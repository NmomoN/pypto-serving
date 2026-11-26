# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Text-only V4.1 A3 configuration independent of PyPTO kernel layouts."""

from pathlib import Path

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.model_family import is_deepseek_v41_config


def validate_checkpoint_config(raw):
    quant = raw.get("quantization_config")
    if (not is_deepseek_v41_config(raw) or not isinstance(quant, dict)
            or quant.get("quant_method") != "ascend"
            or quant.get("model_quant_type") != "W8A8_DYNAMIC"):
        raise ValueError("Torch V4.1 A3 requires a DeepSeek V4.1 ascend/W8A8_DYNAMIC checkpoint")
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Torch V4.1 requires text_config")
    for name in ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
                 "head_dim", "moe_intermediate_size", "max_position_embeddings"):
        if type(text.get(name)) is not int or text[name] <= 0:
            raise ValueError(f"Invalid V4.1 text_config.{name}")
    return text


def build_torch_engine_config(args, model_dir, raw, devices):
    from pypto_serving.cli.main import _build_profile_config
    from pypto_serving.serving.engine.async_engine import EngineConfig

    text = validate_checkpoint_config(raw)
    if args.platform != "a2a3":
        raise ValueError("Torch V4.1 W8A8 requires platform a2a3")
    tp, dp, ep = args.tensor_parallel_size, args.data_parallel_size, args.expert_parallel_size
    if dp < 1 or tp < 1 or dp * tp != len(devices) or ep != (len(devices) if dp > 1 else 1):
        raise ValueError("Torch V4.1 requires TP equal to devices for DP=EP=1, or EP=DP*TP=devices")
    if text["num_attention_heads"] % tp:
        raise ValueError("V4.1 attention heads must be divisible by TP")
    if args.dtype != "bfloat16" or args.kv_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Torch V4.1 requires BF16 activation and KV metadata with W8A8 weights")
    if args.kv_offload_bytes or args.pd_role:
        raise ValueError("Torch V4.1 requires resident colocated KV cache")
    if args.enable_prefix_caching:
        raise ValueError("Torch V4.1 requires --no-enable-prefix-caching for recurrent/Engram state")
    if args.num_speculative_tokens or args.speculative_config:
        raise ValueError("Torch V4.1 initially supports target-only autoregressive decoding")
    parallel = ParallelConfig(
        devices=devices, tensor_parallel_size=tp, data_parallel_size=dp, expert_parallel_size=ep,
        enable_expert_parallel=dp > 1, placement_mode="data_expert" if dp > 1 else "replica",
    )
    runtime = RuntimeConfig(
        device="cpu", page_size=args.block_size, max_batch_size=args.max_num_seqs,
        max_seq_len=args.max_model_len, weight_dtype="bfloat16", kv_dtype="bfloat16",
        max_num_batched_tokens=args.max_num_batched_tokens, total_kv_pages=args.total_kv_pages,
        npu_memory_utilization=args.npu_memory_utilization,
    )
    return EngineConfig(
        model_id=args.served_model_name or Path(model_dir).name,
        model_dir=model_dir, platform=args.platform, device_id=devices[0], device_ids=devices,
        parallel_config=parallel, executor_cls="TorchDeepSeekV41Executor",
        executor_kwargs=dict(graph_mode=args.torch_graph_mode,
                             data_parallel_size=dp,
                             async_scheduling=args.async_scheduling is not False,
                             enable_chunked_prefill=args.enable_chunked_prefill),
        runtime_config=runtime, profile_config=_build_profile_config(args),
        max_num_running_reqs=args.max_num_seqs, max_num_scheduled_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_cache=False, enable_chunk_prefill=args.enable_chunked_prefill,
        async_scheduling=args.async_scheduling,
    )
