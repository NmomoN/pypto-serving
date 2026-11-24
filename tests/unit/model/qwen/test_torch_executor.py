# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression checks for external scheduling and KV ownership in the Torch adapter."""

import json
from types import SimpleNamespace

import pytest

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.model.qwen.torch_executor import TorchQwenExecutor, extend_block_table


def test_external_page_ids_reserve_null_block_and_append_only_new_pages():
    assert extend_block_table((), [0, 9]) == ([1, 10],)
    assert extend_block_table((1, 10), [0, 9]) is None
    assert extend_block_table((1, 10), [0, 9, 4]) == ([5],)


@pytest.mark.parametrize("pages", [[9, 0], [0], [-1, 9]])
def test_page_replacement_requires_request_restart(pages):
    with pytest.raises(ValueError):
        extend_block_table((1, 10), pages)


def test_chunked_registration_retains_complete_prompt_and_restart_is_explicit():
    executor = TorchQwenExecutor()
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(1000)))
    executor.register_requests([request])
    assert executor._requests["r"].prompt_token_ids == list(range(1000))
    executor._started.add("r")
    executor._blocks["r"] = (1, 4)
    restarted = SimpleNamespace(request_id="r", prompt_token_ids=list(range(1010)))
    executor.register_requests([restarted])
    assert executor._finished == {"r"}
    assert "r" not in executor._started
    assert "r" not in executor._blocks
    assert len(executor._requests["r"].prompt_token_ids) == 1010


def test_release_removes_request_state_before_page_reuse():
    executor = TorchQwenExecutor()
    executor.register_requests([SimpleNamespace(request_id="r", prompt_token_ids=[1])])
    executor._started.add("r")
    executor._blocks["r"] = (1,)
    executor.release_finished_requests(["r"])
    assert executor._requests == {}
    assert executor._started == set()
    assert executor._blocks == {}
    assert executor._finished == {"r"}


def parse_config(tmp_path, *extra, architecture="Qwen3ForCausalLM"):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    args = build_parser().parse_args(["--model", str(tmp_path), *extra])
    return build_serving_engine_config(args)


def test_torch_cli_selects_adapter_and_preserves_native_default(tmp_path):
    config = parse_config(tmp_path, "--backend", "torch", "--torch-graph-mode", "FULL")
    assert config.executor_cls == "TorchQwenExecutor"
    assert config.executor_kwargs == {
        "graph_mode": "FULL", "enable_prefix_caching": True, "enable_chunked_prefill": True,
        "async_scheduling": True,
    }
    unchunked = parse_config(tmp_path, "--backend", "torch", "--no-enable-chunked-prefill")
    assert unchunked.executor_kwargs["enable_chunked_prefill"] is False
    native = parse_config(tmp_path)
    assert native.executor_cls == "PyptoQwen14BExecutor"


@pytest.mark.parametrize("options", [
    ("--devices", "0,1", "--tp", "2"),
    ("--dtype", "float16"),
    ("--kv-offload-bytes", "1024"),
])
def test_unsupported_torch_configuration_fails_before_worker_start(tmp_path, options):
    with pytest.raises(ValueError):
        parse_config(tmp_path, "--backend", "torch", *options)


def test_torch_cli_rejects_unknown_architecture_instead_of_qwen_fallback(tmp_path):
    with pytest.raises(ValueError, match="dense Qwen3ForCausalLM"):
        parse_config(tmp_path, "--backend", "torch", architecture="Qwen3MoeForCausalLM")


def test_async_decode_preparation_retains_identity_without_reading_model_state():
    executor = TorchQwenExecutor()
    batch = SimpleNamespace(request_ids=["a", "b"], allow_device_greedy_sampling=True)
    prepared = executor.prepare_decode(None, batch, buffer_slot=0)
    batch.request_ids.reverse()
    assert prepared.request_ids == ("a", "b")
    assert not executor.prepared_decode_requires_token(prepared)
    assert executor.supports_async_decode_prepare
    assert executor.supports_async_decode_reclaim
    assert executor.supports_async_prefill_sampling
    assert executor.supports_device_token_dependency
    with pytest.raises(ValueError, match="request IDs changed"):
        executor.dispatch_prepared_decode(None, batch, prepared)


def test_host_sampling_keeps_the_actual_token_dependency():
    executor = TorchQwenExecutor()
    batch = SimpleNamespace(request_ids=["a"], allow_device_greedy_sampling=False)
    assert executor.prepared_decode_requires_token(executor.prepare_decode(None, batch, buffer_slot=1))
    synchronous = TorchQwenExecutor(async_scheduling=False)
    assert not synchronous.supports_async_decode_reclaim
    assert not synchronous.supports_async_prefill_reclaim


def test_cli_controls_both_scheduler_and_torch_model_runner(tmp_path):
    config = parse_config(tmp_path, "--backend", "torch", "--async-scheduling")
    assert config.resolve_async_scheduling() is True
    assert config.executor_kwargs["async_scheduling"] is True
    sync = parse_config(tmp_path, "--backend", "torch", "--no-async-scheduling")
    assert sync.resolve_async_scheduling() is False
    assert sync.executor_kwargs["async_scheduling"] is False


def _fake_device(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None), raising=False)


def test_async_native_output_is_only_materialized_at_reclaim(monkeypatch):
    import threading
    import torch
    from pypto_serving.model.qwen.torch_executor import _PendingOutput

    _fake_device(monkeypatch)
    called = []
    native = SimpleNamespace()
    def get_output():
        called.append(threading.current_thread().name)
        return SimpleNamespace(req_id_to_index={"b": 0, "a": 1}, sampled_token_ids=[[20], [10]])
    native.get_output = get_output
    executor = TorchQwenExecutor()
    executor.worker = SimpleNamespace(device="npu:0")
    executor._cpu_tokens = {"a": [], "b": []}
    pending = _PendingOutput(("a", "b"), native, torch.empty(2, 0))
    assert not called
    result = []
    thread = threading.Thread(target=lambda: result.append(executor.reclaim_prepared_decode(pending)), name="reclaim")
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert called == ["reclaim"]
    assert result[0].sampled_token_ids.tolist() == [10, 20]
    assert executor._cpu_tokens == {"a": [10], "b": [20]}


def test_reactivated_request_waits_for_its_committed_history(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from pypto_serving.model.qwen.torch_executor import _PendingOutput

    _fake_device(monkeypatch)
    executor = TorchQwenExecutor()
    executor.worker = SimpleNamespace(device="npu:0")
    executor.register_requests([SimpleNamespace(request_id="a", prompt_token_ids=[1, 2])])
    executor._cpu_tokens["a"] = [10, 11]
    native = SimpleNamespace(get_output=lambda: SimpleNamespace(
        req_id_to_index={"a": 0}, sampled_token_ids=[[12]],
    ))
    with ThreadPoolExecutor(max_workers=1) as pool:
        recovered = pool.submit(executor._recovered_token_ids, "a", 3)
        executor.reclaim_prepared_decode(_PendingOutput(("a",), native, torch.empty(1, 0)))
        assert recovered.result(timeout=2) == [1, 2, 10, 11, 12]


def test_failed_reclaim_wakes_reactivation_waiters(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from pypto_serving.model.qwen.torch_executor import _PendingOutput

    _fake_device(monkeypatch)
    executor = TorchQwenExecutor()
    executor.worker = SimpleNamespace(device="npu:0")
    executor.register_requests([SimpleNamespace(request_id="a", prompt_token_ids=[1])])
    def fail():
        raise ValueError("copy failed")
    native = SimpleNamespace(get_output=fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        recovered = pool.submit(executor._recovered_token_ids, "a", 1)
        with pytest.raises(ValueError, match="copy failed"):
            executor.reclaim_prepared_decode(_PendingOutput(("a",), native, torch.empty(1, 0)))
        with pytest.raises(RuntimeError, match="token recovery failed"):
            recovered.result(timeout=2)


def test_async_prefill_uses_the_dispatched_prompt_length_after_reregistration():
    import torch
    from pypto_serving.config.types import PrefillBatch, PrefillResult, SamplingParams
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    worker = WorkerProcess.__new__(WorkerProcess)
    worker._req_cache = {"a": SimpleNamespace(prompt_token_ids=[1] * 999)}
    worker.executor = SimpleNamespace(finalize_prefill=lambda *args: None)
    worker._sampling_params = lambda _: SamplingParams(temperature=0, top_p=1)
    worker._sample_result_row = lambda *args, **kwargs: 17
    worker._accept_output_tokens = lambda *args: None
    batch = PrefillBatch(
        request_ids=["a"], token_ids=torch.tensor([1, 2]), input_embeddings=None,
        seq_lens=[2], chunk_lens=[2], chunk_offsets=[0], chunk_starts=[0], prompt_lens=[2],
    )
    scheduled = [SimpleNamespace(request_id="a", num_computed_tokens=0, chunk_tokens=[1, 2])]
    output = {}
    worker._consume_prefill_result(
        scheduled, SimpleNamespace(extra={}), batch,
        PrefillResult(last_hidden=None, logits=torch.empty(1, 0), sampled_token_ids=torch.tensor([17])),
        output,
    )
    assert output == {"a": [17]}


def test_explicit_cache_capacity_reaches_runtime(tmp_path):
    config = parse_config(tmp_path, "--backend", "torch", "--total-kv-pages", "1380")
    assert config.runtime_config.total_kv_pages == 1380
    with pytest.raises(ValueError, match="must be positive"):
        parse_config(tmp_path, "--backend", "torch", "--total-kv-pages", "0")


def test_failed_submission_wakes_token_recovery_waiters(monkeypatch):
    import contextlib
    from concurrent.futures import ThreadPoolExecutor
    import sys
    import types

    _fake_device(monkeypatch)
    fake_config = types.ModuleType("vllm.config")
    fake_config.set_current_vllm_config = lambda _: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    executor = TorchQwenExecutor()
    executor.config = object()
    executor.register_requests([SimpleNamespace(request_id="a", prompt_token_ids=[1])])
    def fail(_):
        raise RuntimeError("native launch failed")
    executor.worker = SimpleNamespace(device="npu:0", execute_model=fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        recovered = pool.submit(executor._recovered_token_ids, "a", 1)
        with pytest.raises(RuntimeError, match="native launch failed"):
            executor._submit(["a"], object(), device_sampling=True, stop_masks={})
        with pytest.raises(RuntimeError, match="token recovery failed"):
            recovered.result(timeout=2)
