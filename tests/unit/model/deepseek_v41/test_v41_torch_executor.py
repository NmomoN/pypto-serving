# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host contract checks for V4.1 W8A8, hybrid KV ownership and async RPC."""

import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
import torch

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.deepseek_v41.torch_config import validate_checkpoint_config
from pypto_serving.model.deepseek_v41.torch_executor import (
    PendingOutput, TorchDeepSeekV41Executor, apply_stop_masks, logical_page_capacity,
)
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.serving.server.ipc import NewRequestData


def raw_config():
    return dict(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"],
                quantization_config=dict(quant_method="ascend", model_quant_type="W8A8_DYNAMIC"),
                text_config=dict(hidden_size=8, vocab_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, head_dim=4, moe_intermediate_size=16,
                                 max_position_embeddings=256, rope_theta=10000))


def config(tmp_path, *extra):
    (tmp_path / "config.json").write_text(json.dumps(raw_config()))
    args = build_parser().parse_args([
        "--model", str(tmp_path), "--backend", "torch", "--no-enable-prefix-caching",
        "--max-model-len", "128", "--block-size", "64", *extra])
    return build_serving_engine_config(args)


def request(name="r", prompt=(1, 2, 3, 4)):
    return NewRequestData(name, list(prompt), 0.0, 1.0, None)


class Blocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self, allow_none=False):
        return self.ids


class PhysicalKV:
    def __init__(self):
        self.calls = []
        self.freed = []
        self.exhausted = False

    def allocate_slots(self, req, num_new_tokens):
        self.calls.append((req.request_id, req.num_computed_tokens, num_new_tokens))
        return None if self.exhausted else Blocks(([7], [21]))

    def get_blocks(self, rid):
        return Blocks(([7, 8], [21, 22]))

    def take_new_block_ids(self):
        return [7, 21]

    def free(self, req):
        self.freed.append(req.request_id)


class RPC:
    def __init__(self):
        self.calls = []
        self.executions = []
        self.samplings = []

    def execute_model(self, schedule, non_block=False):
        self.calls.append(("execute", schedule, non_block))
        if not non_block:
            return None
        future = Future()
        self.executions.append(future)
        return future

    def sample_tokens(self, grammar, non_block=False):
        self.calls.append(("sample", grammar, non_block))
        future = Future()
        self.samplings.append(future)
        return future

    def shutdown(self):
        self.calls.append(("shutdown",))


def executor():
    result = TorchDeepSeekV41Executor(device_ids=(1, 3))
    result.runtime = RuntimeConfig(max_batch_size=2, max_seq_len=128)
    result.native_kv = PhysicalKV()
    result.worker = RPC()
    result._group_count = 2
    result._zero_blocks = True
    return result


def test_cli_uses_one_serving_scheduler_and_native_tp_group(tmp_path):
    result = config(tmp_path, "--devices", "1,3", "--tp", "2", "--async-scheduling")
    assert result.executor_cls == "TorchDeepSeekV41Executor"
    assert result.worker_device_ids() == (1, 3)
    assert result.parallel_config.num_replicas == 1
    assert result.runtime_config.kv_cache_groups == ()
    assert not result.enable_prefix_cache
    assert result.executor_kwargs["async_scheduling"] is True
    assert result.sampling_capabilities().stop_mask_on_device("decode")


@pytest.mark.parametrize("options,match", [
    (("--enable-prefix-caching",), "prefix"),
    (("--platform", "a5"), "a2a3"),
    (("--devices", "0,1", "--tp", "1"), "TP equal"),
    (("--dtype", "float16"), "BF16"),
    (("--kv-offload-bytes", "1024"), "resident"),
    (("--num-speculative-tokens", "1"), "target-only"),
])
def test_invalid_configuration_fails_before_model_payloads(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        config(tmp_path, *options)


def test_v4_compressed_tensors_and_v41_fp8_do_not_pass_as_a3_w8a8():
    raw = raw_config()
    raw["quantization_config"]["quant_method"] = "compressed-tensors"
    with pytest.raises(ValueError, match="W8A8_DYNAMIC"):
        validate_checkpoint_config(raw)
    raw = raw_config()
    raw["model_type"] = "deepseek_v4"
    raw["architectures"] = ["DeepseekV4ForCausalLM"]
    with pytest.raises(ValueError, match="V4.1"):
        validate_checkpoint_config(raw)


def test_metadata_loader_skips_lib_runtime_and_never_materializes_payload(tmp_path, monkeypatch):
    from pypto_serving.model.deepseek_v41 import torch_loader

    (tmp_path / "config.json").write_text(json.dumps(raw_config()))
    shard = tmp_path / "quant.safetensors"
    shard.write_bytes(b"not opened by metadata loader")
    keys = ["embed.weight", "head.weight", "norm.weight"]
    keys += ["layers.0.attn.wq_a." + s for s in ("weight", "weight_scale", "weight_offset")]
    index_path = tmp_path / "quant_model_weights.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {k: shard.name for k in keys}}))
    monkeypatch.setattr(torch_loader, "load_tokenizer", lambda path: NS(
        bos_token_id=1, eos_token_id=2, pad_token_id=0))
    import sys
    monkeypatch.setitem(sys.modules, "pypto.runtime", None)
    loader = ModelLoader(format_loaders=[torch_loader.DeepSeekV41TorchDirectoryLoader()])
    loaded = loader.load("m", str(tmp_path), model_format="deepseek_v41_torch")
    assert loaded.runtime_model.embed_tokens.numel() == 0
    assert loaded.runtime_model.extra["checkpoint_format"] == "ascend-w8a8-dynamic"
    shard.unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        loader.load("m", str(tmp_path), model_format="deepseek_v41_torch")


def test_capacity_accounts_for_all_groups_not_only_primary():
    cache = NS(num_blocks=99, kv_cache_groups=[
        NS(kv_cache_spec=NS(block_size=64)), NS(kv_cache_spec=NS(block_size=32))])
    runtime = RuntimeConfig(page_size=64, max_batch_size=2)
    assert logical_page_capacity(cache, runtime) == 30
    with pytest.raises(ValueError, match="budget"):
        logical_page_capacity(cache, RuntimeConfig(page_size=64, max_batch_size=2, total_kv_pages=31))


def test_hybrid_group_ids_come_from_native_allocator_and_prompt_is_complete():
    e = executor()
    e.register_requests([request()])
    output = e._schedule(["r"], [2], [0])
    assert output.scheduled_new_reqs[0].prompt_token_ids == [1, 2, 3, 4]
    assert output.scheduled_new_reqs[0].block_ids == ([7, 8], [21, 22])
    assert output.new_block_ids_to_zero == [7, 21]
    e._previous_ids = {"r"}
    next_output = e._schedule(["r"], [2], [2])
    assert next_output.scheduled_cached_reqs.new_block_ids == [([7], [21])]
    assert next_output.scheduled_cached_reqs.num_output_tokens == [0]


def test_device_submission_does_not_wait_for_sampling_or_cpu_tokens():
    e = executor()
    e.register_requests([request()])
    first = e._submit(["r"], [4], [0], device_sampling=True, stop_masks={})
    second = e._submit(["r"], [1], [4], device_sampling=True, stop_masks={})
    assert [x[0] for x in e.worker.calls] == ["execute", "sample", "execute", "sample"]
    assert not first.sampling.done() and not second.sampling.done()
    assert e.worker.calls[2][1].scheduled_cached_reqs.num_output_tokens == [1]


def test_reclaim_reorders_native_rows_and_publishes_cpu_history():
    e = executor()
    e.register_requests([request("a"), request("b")])
    execution, sampling = Future(), Future()
    execution.set_result(None)
    sampling.set_result(NS(req_id_to_index={"a": 1, "b": 0}, sampled_token_ids=[[12], [31]]))
    _, tokens = e._reclaim(PendingOutput(("a", "b"), execution, sampling))
    assert tokens.tolist() == [31, 12]
    assert e._cpu_tokens == {"a": [31], "b": [12]}


def test_rejoining_batch_recovers_real_history_and_failure_wakes_waiter():
    e = executor()
    e.register_requests([request()])
    e._schedule(["r"], [4], [0])
    e._cpu_tokens["r"] = [17]
    output = e._schedule(["r"], [1], [4])
    assert output.scheduled_cached_reqs.all_token_ids["r"] == [1, 2, 3, 4, 17]
    caught = []
    def recover():
        try:
            e._history("r", 2)
        except RuntimeError as error:
            caught.append(str(error))
    thread = threading.Thread(target=recover)
    thread.start()
    e._fail(RuntimeError("native RPC failed"))
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert caught == ["Asynchronous V4.1 output failed"]


def test_release_and_restart_free_native_state_and_emit_finished_ids():
    e = executor()
    e.register_requests([request()])
    e._schedule(["r"], [4], [0])
    e.release_finished_requests(["r"])
    assert e.native_kv.freed == ["r"]
    assert e.worker.calls[-1][1].finished_req_ids == {"r"}
    assert not e._native_requests and not e._requests
    e.register_requests([request()])
    e._schedule(["r"], [4], [0])
    e.register_requests([request(prompt=(8, 9))])
    output = e._schedule(["r"], [2], [0])
    assert output.finished_req_ids == {"r"}
    assert output.scheduled_new_reqs[0].prompt_token_ids == [8, 9]


def test_physical_capacity_failure_is_explicit_and_prevents_further_dispatch():
    e = executor()
    e.register_requests([request()])
    e.native_kv.exhausted = True
    with pytest.raises(RuntimeError, match="exhausted"):
        e._submit(["r"], [4], [0], device_sampling=True, stop_masks={})
    assert not e.worker.calls
    assert e._async_error is not None


def test_stop_mask_is_applied_on_native_logits():
    logits = torch.zeros(2, 32)
    worker = NS(model_runner=NS(execute_model_state=NS(logits=logits),
                               input_batch=NS(req_id_to_index={"b": 0, "a": 1})))
    apply_stop_masks(worker, {"a": NS(remaining_tokens=1, stop_token_ids=(3, 7))})
    assert torch.isneginf(logits[1, [3, 7]]).all()
    assert not torch.isneginf(logits[0]).any()


def test_old_native_stack_is_rejected_before_loading_weights(monkeypatch):
    from pypto_serving.model.deepseek_v41 import torch_executor

    monkeypatch.setattr(torch_executor, "version", lambda package: "0.23.0")
    with pytest.raises(ValueError, match="0.30"):
        TorchDeepSeekV41Executor().register_model("m", None)


def test_startup_uses_all_native_rank_configs_and_cleans_up_on_failure(monkeypatch):
    from pypto_serving.model.deepseek_v41 import torch_executor
    import vllm.engine.arg_utils as args_module
    import vllm.plugins as plugins
    import vllm.v1.executor.abstract as abstract
    import vllm.v1.core.kv_cache_manager as manager
    import vllm.v1.core.kv_cache_utils as utils

    captured = {}
    cache = NS(num_blocks=99, needs_kv_cache_zeroing=True, kv_cache_groups=[
        NS(kv_cache_spec=NS(block_size=64)), NS(kv_cache_spec=NS(block_size=32))])
    native_config = NS(cache_config=NS(num_gpu_blocks=None), max_in_flight_tokens=8)
    class Arguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def create_engine_config(self):
            return native_config
    class Native(RPC):
        def __init__(self, config):
            super().__init__()
            self.initialized = []
        def get_kv_cache_specs(self):
            return [{"rank": 0}, {"rank": 1}]
        def determine_available_memory(self):
            return [1000, 1001]
        def initialize_from_config(self, caches):
            self.initialized = caches
    native = Native(native_config)
    monkeypatch.setattr(torch_executor, "version", lambda package: "0.30.0rc2")
    monkeypatch.setattr(plugins, "load_general_plugins", lambda: None)
    monkeypatch.setattr(args_module, "EngineArgs", Arguments)
    monkeypatch.setattr(abstract.Executor, "get_class", lambda config: lambda config: native)
    monkeypatch.setattr(utils, "get_kv_cache_configs", lambda config, specs, memory: [cache, cache])
    monkeypatch.setattr(utils, "generate_scheduler_kv_cache_config", lambda configs: configs[0])
    monkeypatch.setattr(utils, "resolve_kv_cache_block_sizes", lambda cache, config: (64, 64))
    monkeypatch.setattr(manager, "KVCacheManager", lambda *args, **kwargs: NS())
    e = TorchDeepSeekV41Executor(device_ids=(1, 3))
    record = NS(config=NS(architecture="DeepseekV41ForCausalLM"),
                runtime=RuntimeConfig(page_size=64, max_batch_size=2, max_seq_len=128),
                runtime_model=NS(extra=dict(model_dir="/checkpoint", checkpoint_format="ascend-w8a8-dynamic")))
    assert e.register_model("m", record) == 30
    assert native.initialized == [cache, cache]
    assert captured["tensor_parallel_size"] == 2
    assert captured["distributed_executor_backend"] == "mp"
    assert captured["quantization"] == "ascend"
    assert not captured["enable_prefix_caching"]
    e.close()
    assert native.calls[-1] == ("shutdown",)
    monkeypatch.setattr(native, "initialize_from_config",
                        lambda caches: (_ for _ in ()).throw(RuntimeError("init failed")))
    with pytest.raises(RuntimeError, match="init failed"):
        e.register_model("m", record)
    assert native.calls[-1] == ("shutdown",)
    assert e.worker is None


def test_control_rpc_and_reclaim_share_one_fifo_result_consumer():
    import time

    e = executor()
    e.register_requests([request("done"), request("b")])
    e._started.add("done")
    active = 0
    peak = 0
    guard = threading.Lock()
    entered = threading.Event()
    class Guarded:
        def __init__(self, value):
            self.value = value
        def result(self):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            entered.set()
            time.sleep(0.02)
            with guard:
                active -= 1
            return self.value
    class ControlRPC(RPC):
        def execute_model(self, schedule, non_block=False):
            return Guarded(None).result()
    e.worker = ControlRPC()
    output = NS(req_id_to_index={"b": 0}, sampled_token_ids=[[9]])
    pending = PendingOutput(("b",), Guarded(None), Guarded(output))
    errors = []
    def reclaim():
        try:
            e._reclaim(pending)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=reclaim)
    thread.start()
    assert entered.wait(timeout=2)
    e.release_finished_requests(["done"])
    thread.join(timeout=2)
    assert not thread.is_alive() and not errors
    assert peak == 1
    assert e._cpu_tokens["b"] == [9]
