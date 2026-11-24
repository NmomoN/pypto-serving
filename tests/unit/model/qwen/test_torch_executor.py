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
