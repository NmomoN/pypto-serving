# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run serving-owned Qwen batches through the vLLM Ascend Torch worker."""

from __future__ import annotations

import logging
import os
import socket
from importlib.metadata import version

import torch

from pypto_serving.config.types import DecodeResult, PrefillResult
from pypto_serving.model.common.executor.executor import ModelExecutor

logger = logging.getLogger(__name__)


def extend_block_table(previous: tuple[int, ...], current: list[int]) -> tuple[list[int], ...] | None:
    """Translate serving page IDs into a vLLM append, reserving its null page."""
    translated = tuple(page + 1 for page in current)
    if any(page < 0 for page in current) or translated[:len(previous)] != previous:
        raise ValueError("KV pages changed without a request restart")
    return (list(translated[len(previous):]),) if len(translated) > len(previous) else None


class TorchQwenExecutor(ModelExecutor):
    """Keep scheduling in Serving and reuse vLLM's model, attention, and sampler.

    The adapter targets vLLM/Ascend 0.23.0, one dense Qwen3 model and one NPU per
    worker. Serving supplies token counts and KV page assignments while the
    vLLM worker owns model execution and sampling.
    """

    def __init__(
        self, kv_cache_manager=None, *, platform="a2a3", device_ids=(0,),
        pypto_build_dir=None, graph_mode="NONE", enable_prefix_caching=True,
        enable_chunked_prefill=True,
    ):
        super().__init__(kv_cache_manager)
        if platform != "a2a3" or len(device_ids) != 1:
            raise ValueError("Torch Qwen currently requires platform a2a3 and one NPU per worker")
        if graph_mode not in ("NONE", "FULL", "PIECEWISE"):
            raise ValueError(f"Unsupported Torch graph mode {graph_mode!r}")
        self.device_id = int(device_ids[0])
        self.graph_mode = graph_mode
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.worker = None
        self.config = None
        self.runtime = None
        self._requests = {}
        self._started = set()
        self._blocks = {}
        self._finished = set()

    @property
    def supports_device_embedding(self):
        return True

    @property
    def supports_device_sampling(self):
        return True

    @property
    def supports_device_stochastic_sampling(self):
        return True

    @property
    def max_prefill_requests_per_partition(self):
        return self.runtime.max_batch_size if self.runtime else 1

    def register_model(self, model_id, record):
        """Initialize the same NPUWorker and cache configuration as native vLLM."""
        if record.config.architecture != "Qwen3ForCausalLM":
            raise ValueError("Torch Qwen requires a dense Qwen3ForCausalLM checkpoint")
        if record.runtime.weight_dtype != "bfloat16" or record.runtime.kv_dtype != "bfloat16":
            raise ValueError("Torch Qwen currently requires BF16 weights and KV cache")
        if record.runtime.kv_cache_groups or record.runtime.kv_offload_bytes:
            raise ValueError("Torch Qwen currently uses a single resident KV cache pool")
        for package in ("vllm", "vllm-ascend"):
            if version(package).split("+")[0] != "0.23.0":
                raise ValueError(f"Torch Qwen requires {package} 0.23.0")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device_id)
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import set_current_vllm_config
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm_ascend.worker.worker import NPUWorker

        self.runtime = record.runtime
        args = EngineArgs(
            model=record.runtime_model.extra["model_dir"], dtype="bfloat16",
            tensor_parallel_size=1, max_model_len=self.runtime.max_seq_len,
            max_num_seqs=self.runtime.max_batch_size,
            max_num_batched_tokens=self.runtime.max_num_batched_tokens,
            block_size=self.runtime.page_size,
            gpu_memory_utilization=self.runtime.npu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_chunked_prefill=self.enable_chunked_prefill,
            enforce_eager=self.graph_mode == "NONE", async_scheduling=False,
            compilation_config={"cudagraph_mode": self.graph_mode},
            generation_config="vllm", seed=0,
        )
        self.config = args.create_engine_config()
        if self.runtime.total_kv_pages is not None:
            self.config.cache_config.num_gpu_blocks_override = self.runtime.total_kv_pages + 1
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        try:
            with set_current_vllm_config(self.config):
                self.worker = NPUWorker(
                    self.config, local_rank=0, rank=0,
                    distributed_init_method=f"tcp://127.0.0.1:{port}", is_driver_worker=True,
                )
                self.worker.init_device()
                self.worker.load_model()
                available = self.worker.determine_available_memory()
                specs = self.worker.get_kv_cache_spec()
                cache = get_kv_cache_configs(self.config, [specs], [available])[0]
                if len(cache.kv_cache_groups) != 1 or cache.num_blocks < 2:
                    raise ValueError("Torch Qwen requires one full-attention KV group with at least two pages")
                self.config.cache_config.num_gpu_blocks = cache.num_blocks
                self.worker.initialize_from_config(cache)
                self.worker.compile_or_warm_up_model()
            logger.info(
                "Torch backend model=%s runner=%s graph=%s physical_device=%s pages=%s "
                "block_size=%s; Serving owns scheduling and page allocation",
                type(self.worker.model_runner.get_model()).__module__,
                type(self.worker.model_runner).__module__, self.graph_mode,
                self.device_id, cache.num_blocks - 1, self.runtime.page_size,
            )
            return cache.num_blocks - 1
        except BaseException:
            self.close()
            raise

    def register_requests(self, requests):
        """Retain complete prompts before the first possibly chunked prefill."""
        for request in requests:
            if request.request_id in self._started:
                self._finished.add(request.request_id)
            self._requests[request.request_id] = request
            self._started.discard(request.request_id)
            self._blocks.pop(request.request_id, None)

    def release_finished_requests(self, request_ids):
        for request_id in request_ids:
            if request_id in self._started:
                self._finished.add(request_id)
            self._requests.pop(request_id, None)
            self._started.discard(request_id)
            self._blocks.pop(request_id, None)
        if self.worker is not None and self._finished:
            from vllm.config import set_current_vllm_config
            from vllm.v1.core.sched.output import SchedulerOutput

            output = SchedulerOutput.make_empty()
            output.finished_req_ids = self._finished.copy()
            with torch.inference_mode(), set_current_vllm_config(self.config):
                self.worker.execute_model(output)
            self._finished.clear()

    def _schedule(self, request_ids, token_counts, computed_tokens, block_ids):
        from vllm import SamplingParams
        from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

        output = SchedulerOutput.make_empty()
        output.finished_req_ids = self._finished.copy()
        self._finished.clear()
        output.num_common_prefix_blocks = [0]
        for request_id, count, computed, blocks in zip(
            request_ids, token_counts, computed_tokens, block_ids, strict=True,
        ):
            delta = extend_block_table(self._blocks.get(request_id, ()), blocks)
            if request_id not in self._started:
                request = self._requests[request_id]
                params = SamplingParams(
                    temperature=request.temperature, top_p=request.top_p,
                    top_k=request.top_k if request.top_k is not None and request.top_k > 0 else -1,
                    seed=request.seed, max_tokens=self.runtime.max_seq_len, ignore_eos=True,
                )
                output.scheduled_new_reqs.append(NewRequestData(
                    req_id=request_id, prompt_token_ids=list(request.prompt_token_ids),
                    mm_features=[], sampling_params=params, pooling_params=None,
                    block_ids=delta or ([],), num_computed_tokens=int(computed), lora_request=None,
                ))
                self._started.add(request_id)
            else:
                cached = output.scheduled_cached_reqs
                cached.req_ids.append(request_id)
                cached.new_block_ids.append(delta)
                cached.num_computed_tokens.append(int(computed))
                state = self.worker.model_runner.requests[request_id]
                cached.num_output_tokens.append(len(state.output_token_ids))
            self._blocks[request_id] = tuple(page + 1 for page in blocks)
            output.num_scheduled_tokens[request_id] = int(count)
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        return output

    def _execute(self, request_ids, schedule, *, device_sampling, stop_masks):
        from vllm.config import set_current_vllm_config

        with torch.inference_mode(), set_current_vllm_config(self.config):
            torch.npu.set_device(self.worker.device)
            immediate = self.worker.execute_model(schedule)
            if immediate is not None:
                raise RuntimeError("Expected deferred logits from the synchronous vLLM model runner")
            runner = self.worker.model_runner
            state = runner.execute_model_state
            logits = state.logits
            for request_id, mask in stop_masks.items():
                if mask.remaining_tokens > 0:
                    row = runner.input_batch.req_id_to_index[request_id]
                    logits[row, list(mask.stop_token_ids)] = -torch.inf
            if device_sampling:
                host_logits = torch.empty((len(request_ids), 0))
            else:
                rows = [runner.input_batch.req_id_to_index[request_id] for request_id in request_ids]
                host_logits = logits[rows].float().cpu()
            result = self.worker.sample_tokens(None)
            tokens = []
            for request_id in request_ids:
                index = result.req_id_to_index[request_id]
                sampled = result.sampled_token_ids[index]
                tokens.append(sampled[0] if sampled else -1)
            return host_logits, torch.tensor(tokens, dtype=torch.long)

    def run_prefill(self, model, batch):
        schedule = self._schedule(
            batch.request_ids, batch.chunk_lens, batch.chunk_starts, batch.block_ids,
        )
        has_terminal_prompt = any(
            start + count >= length for start, count, length in zip(
                batch.chunk_starts, batch.chunk_lens, batch.prompt_lens, strict=True,
            )
        )
        logits, tokens = self._execute(
            batch.request_ids, schedule,
            device_sampling=batch.allow_device_greedy_sampling or not has_terminal_prompt,
            stop_masks=batch.stop_token_masks,
        )
        return PrefillResult(last_hidden=None, logits=logits, sampled_token_ids=tokens)

    def run_decode(self, model, batch):
        # Host sampling/constraints can replace the runner's speculative sampled
        # token. Feed the actual committed Serving token back before execution.
        runner = self.worker.model_runner
        lengths = batch.seq_lens.tolist()
        for request_id, token, length in zip(
            batch.request_ids, batch.token_ids.view(-1).tolist(), lengths, strict=True,
        ):
            state = runner.requests[request_id]
            if state.output_token_ids and state.output_token_ids[-1] != token:
                state.output_token_ids[-1] = token
                row = runner.input_batch.req_id_to_index.get(request_id)
                if row is not None:
                    runner.input_batch.token_ids_cpu[row, int(length) - 1] = token
        schedule = self._schedule(
            batch.request_ids, [1] * len(batch.request_ids),
            [length - 1 for length in lengths], batch.block_ids,
        )
        logits, tokens = self._execute(
            batch.request_ids, schedule, device_sampling=batch.allow_device_greedy_sampling,
            stop_masks=batch.stop_token_masks,
        )
        return DecodeResult(hidden_states=None, logits=logits, sampled_token_ids=tokens)

    def close(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            try:
                worker.shutdown()
            finally:
                from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

                destroy_model_parallel()
                destroy_distributed_environment()
        self._requests.clear()
        self._started.clear()
        self._blocks.clear()
        self._finished.clear()
