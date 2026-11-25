# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Serving scheduling with vLLM 0.30 V4.1 A3 execution and physical hybrid KV."""

import logging
import math
import os
import threading
from dataclasses import dataclass
from importlib.metadata import version

import torch

from pypto_serving.config.types import DecodeResult, PrefillResult
from pypto_serving.model.common.executor.executor import ModelExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDecode:
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class PendingOutput:
    request_ids: tuple[str, ...]
    execution: object
    sampling: object


def logical_page_capacity(cache, runtime):
    """Conservatively charge every group and reserve per-request boundary slack."""
    groups = cache.kv_cache_groups
    if not groups:
        raise ValueError("V4.1 requires native KV cache groups")
    cost = sum(math.ceil(runtime.page_size / group.kv_cache_spec.block_size) for group in groups)
    # All native groups share physical IDs, unlike independent Lib cache pools.
    reserve = 1 + 2 * runtime.max_batch_size * len(groups)
    capacity = (cache.num_blocks - reserve) // cost
    requested = runtime.total_kv_pages
    if capacity <= 0 or (requested is not None and (requested <= 0 or requested > capacity)):
        raise ValueError("Native KV pool cannot fit the serving budget; reduce sequences/context or KV pages")
    return capacity if requested is None else requested


def apply_stop_masks(worker, masks):
    """Execute on native ranks without copying logits back to Serving."""
    state = worker.model_runner.execute_model_state
    if state is None or state.logits is None:
        return
    rows = worker.model_runner.input_batch.req_id_to_index
    for request_id, mask in masks.items():
        if mask.remaining_tokens > 0:
            state.logits[rows[request_id], list(mask.stop_token_ids)] = -torch.inf


class TorchDeepSeekV41Executor(ModelExecutor):
    """Translate Serving steps into executor RPC, without a native scheduler.

    vLLM's hybrid KV manager owns physical SWA/MLA/state allocations; Serving
    owns admission, a conservative logical budget, batching, IPC and HTTP.
    """

    def __init__(self, kv_cache_manager=None, *, platform="a2a3", device_ids=(0,),
                 pypto_build_dir=None, graph_mode="NONE", async_scheduling=True,
                 enable_chunked_prefill=True):
        super().__init__(kv_cache_manager)
        if platform != "a2a3" or not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("Torch V4.1 requires distinct A2/A3 devices")
        if graph_mode not in ("NONE", "FULL", "PIECEWISE"):
            raise ValueError(f"Unsupported Torch graph mode {graph_mode!r}")
        self.device_ids = tuple(device_ids)
        self.graph_mode = graph_mode
        self.async_scheduling = async_scheduling
        self.enable_chunked_prefill = enable_chunked_prefill
        self.worker = self.config = self.runtime = self.native_kv = None
        self._requests = {}
        self._native_requests = {}
        self._cpu_tokens = {}
        self._started = set()
        self._previous_ids = set()
        self._finished = set()
        self._outputs_ready = threading.Condition()
        # MultiprocExecutor futures share one FIFO response queue.
        self._rpc_read_lock = threading.Lock()
        self._async_error = None

    supports_device_embedding = property(lambda self: True)
    supports_device_sampling = property(lambda self: True)
    supports_device_stochastic_sampling = property(lambda self: True)
    supports_device_token_dependency = property(lambda self: self.async_scheduling)
    supports_async_decode_prepare = property(lambda self: self.async_scheduling)
    supports_async_decode_reclaim = property(lambda self: self.async_scheduling)
    supports_async_prefill_reclaim = property(lambda self: self.async_scheduling)
    supports_async_prefill_sampling = property(lambda self: self.async_scheduling)

    @property
    def max_prefill_requests_per_partition(self):
        return self.runtime.max_batch_size if self.runtime else 1

    def register_model(self, model_id, record):
        for package in ("vllm", "vllm-ascend"):
            if version(package).split("+")[0].split(".")[:2] != ["0", "30"]:
                raise ValueError("Torch V4.1 requires the vLLM/vLLM Ascend 0.30 stack")
        from vllm.engine.arg_utils import EngineArgs
        from vllm.plugins import load_general_plugins
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (
            generate_scheduler_kv_cache_config, get_kv_cache_configs, resolve_kv_cache_block_sizes,
        )

        if (record.config.architecture != "DeepseekV41ForCausalLM"
                or record.runtime_model.extra.get("checkpoint_format") != "ascend-w8a8-dynamic"):
            raise ValueError("Torch V4.1 requires the A3 W8A8 metadata loader")
        if record.runtime.kv_cache_groups or record.runtime.kv_offload_bytes:
            raise ValueError("Torch V4.1 uses native hybrid KV; Lib layouts/offload are unsupported")
        self.runtime = record.runtime
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, self.device_ids))
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        load_general_plugins()
        args = EngineArgs(
            model=record.runtime_model.extra["model_dir"], dtype="bfloat16", quantization="ascend",
            tensor_parallel_size=len(self.device_ids), distributed_executor_backend="mp",
            max_model_len=self.runtime.max_seq_len, max_num_seqs=self.runtime.max_batch_size,
            max_num_batched_tokens=self.runtime.max_num_batched_tokens,
            block_size=self.runtime.page_size, gpu_memory_utilization=self.runtime.npu_memory_utilization,
            enable_prefix_caching=False, enable_chunked_prefill=self.enable_chunked_prefill,
            enforce_eager=self.graph_mode == "NONE", async_scheduling=self.async_scheduling,
            compilation_config={"cudagraph_mode": self.graph_mode},
            generation_config="vllm", limit_mm_per_prompt={"image": 0}, seed=0,
        )
        self.config = args.create_engine_config()
        try:
            self.worker = Executor.get_class(self.config)(self.config)
            specs = self.worker.get_kv_cache_specs()
            available = self.worker.determine_available_memory()
            caches = get_kv_cache_configs(self.config, specs, available)
            scheduler_cache = generate_scheduler_kv_cache_config(caches)
            logical_pages = logical_page_capacity(scheduler_cache, self.runtime)
            self.config.cache_config.num_gpu_blocks = scheduler_cache.num_blocks
            self.worker.initialize_from_config(caches)
            block_size, hash_size = resolve_kv_cache_block_sizes(scheduler_cache, self.config)
            self.native_kv = KVCacheManager(
                scheduler_cache, max_model_len=self.runtime.max_seq_len,
                scheduler_block_size=block_size, hash_block_size=hash_size,
                max_in_flight_tokens=self.config.max_in_flight_tokens, enable_caching=False,
            )
            self._group_count = len(scheduler_cache.kv_cache_groups)
            self._zero_blocks = scheduler_cache.needs_kv_cache_zeroing
            logger.info("Torch V4.1 vLLM=0.30 TP=%d devices=%s graph=%s async=%s "
                        "native_physical_blocks=%d groups=%d serving_logical_pages=%d prefix_cache=False",
                        len(self.device_ids), self.device_ids, self.graph_mode, self.async_scheduling,
                        scheduler_cache.num_blocks, self._group_count, logical_pages)
            return logical_pages
        except BaseException:
            self.close()
            raise

    def _fail(self, error):
        with self._outputs_ready:
            self._async_error = error
            self._outputs_ready.notify_all()

    def register_requests(self, requests):
        for request in requests:
            if request.constraint_spec is not None:
                raise ValueError("Torch V4.1 structured constraints require a native grammar bridge")
            request_id = request.request_id
            if request_id in self._started:
                self._finished.add(request_id)
            native = self._native_requests.pop(request_id, None)
            if native is not None:
                self.native_kv.free(native)
            self._requests[request_id] = request
            self._started.discard(request_id)
            with self._outputs_ready:
                self._cpu_tokens[request_id] = []

    def release_finished_requests(self, request_ids):
        for request_id in request_ids:
            if request_id in self._started:
                self._finished.add(request_id)
            native = self._native_requests.pop(request_id, None)
            if native is not None:
                self.native_kv.free(native)
            self._requests.pop(request_id, None)
            self._started.discard(request_id)
            self._previous_ids.discard(request_id)
            with self._outputs_ready:
                self._cpu_tokens.pop(request_id, None)
        if self.worker is not None and self._finished:
            from vllm.v1.core.sched.output import SchedulerOutput
            output = SchedulerOutput.make_empty()
            output.finished_req_ids = self._finished.copy()
            with self._rpc_read_lock:
                self.worker.execute_model(output, non_block=False)
            self._finished.clear()

    def prepare_decode(self, model, batch, *, buffer_slot):
        return PreparedDecode(tuple(batch.request_ids))

    def prepared_decode_requires_token(self, prepared):
        return False

    def prepare_prefill(self, model, batch, *, buffer_slot):
        return tuple(batch.request_ids)

    def _history(self, request_id, count):
        with self._outputs_ready:
            self._outputs_ready.wait_for(
                lambda: self._async_error is not None or len(self._cpu_tokens[request_id]) >= count)
            if self._async_error is not None:
                raise RuntimeError("Asynchronous V4.1 output failed") from self._async_error
            return list(self._requests[request_id].prompt_token_ids) + self._cpu_tokens[request_id][:count]

    def _schedule(self, request_ids, counts, computed):
        from vllm import SamplingParams
        from vllm.v1.request import Request
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        if self._async_error is not None:
            raise RuntimeError("An earlier V4.1 output failed") from self._async_error
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = self._finished.copy()
        self._finished.clear()
        output.num_common_prefix_blocks = [0] * self._group_count
        for request_id, count, offset in zip(request_ids, counts, computed, strict=True):
            source = self._requests[request_id]
            new = request_id not in self._started
            if new:
                params = SamplingParams(
                    temperature=source.temperature, top_p=source.top_p,
                    top_k=source.top_k if source.top_k is not None and source.top_k > 0 else -1,
                    seed=source.seed, max_tokens=self.runtime.max_seq_len, ignore_eos=True,
                )
                self._native_requests[request_id] = Request(
                    request_id, list(source.prompt_token_ids), params, None)
            native = self._native_requests[request_id]
            expected = max(0, int(offset) - len(source.prompt_token_ids) + 1)
            if len(native.output_token_ids) < expected:
                native.append_output_token_ids([-1] * (expected - len(native.output_token_ids)))
            native.num_computed_tokens = int(offset)
            allocation = self.native_kv.allocate_slots(native, num_new_tokens=int(count))
            if allocation is None:
                raise RuntimeError("Native hybrid KV pool exhausted; reduce sequences or logical pages")
            if new:
                output.scheduled_new_reqs.append(NewRequestData(
                    req_id=request_id, prompt_token_ids=list(source.prompt_token_ids), mm_features=[],
                    sampling_params=native.sampling_params, pooling_params=None,
                    block_ids=self.native_kv.get_blocks(request_id).get_block_ids(),
                    num_computed_tokens=int(offset), lora_request=None,
                ))
                self._started.add(request_id)
            else:
                cached = output.scheduled_cached_reqs
                cached.req_ids.append(request_id)
                cached.new_block_ids.append(allocation.get_block_ids(allow_none=True))
                cached.num_computed_tokens.append(int(offset))
                cached.num_output_tokens.append(expected)
                if self.async_scheduling and request_id not in self._previous_ids and expected:
                    cached.all_token_ids[request_id] = self._history(request_id, expected)
            output.num_scheduled_tokens[request_id] = int(count)
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        new_blocks = self.native_kv.take_new_block_ids()
        output.new_block_ids_to_zero = new_blocks if self._zero_blocks else None
        return output

    def _submit(self, request_ids, counts, computed, *, device_sampling, stop_masks):
        if not device_sampling:
            raise ValueError("Torch V4.1 requires native sampling; host-logit policies are unsupported")
        try:
            schedule = self._schedule(request_ids, counts, computed)
            execution = self.worker.execute_model(schedule, non_block=True)
            if stop_masks:
                with self._rpc_read_lock:
                    self.worker.collective_rpc(apply_stop_masks, args=(stop_masks,))
            sampling = self.worker.sample_tokens(None, non_block=True)
            self._previous_ids = set(request_ids)
            return PendingOutput(tuple(request_ids), execution, sampling)
        except BaseException as error:
            self._fail(error)
            raise

    def _reclaim(self, pending):
        try:
            with self._rpc_read_lock:
                if pending.execution.result() is not None:
                    raise RuntimeError("Expected a deferred native sampling result")
                output = pending.sampling.result()
                if callable(getattr(output, "get_output", None)):
                    output = output.get_output()
            tokens = []
            with self._outputs_ready:
                for request_id in pending.request_ids:
                    sampled = output.sampled_token_ids[output.req_id_to_index[request_id]]
                    tokens.append(sampled[0] if sampled else -1)
                    if request_id in self._cpu_tokens:
                        self._cpu_tokens[request_id].extend(sampled)
                self._outputs_ready.notify_all()
            return torch.empty((len(tokens), 0)), torch.tensor(tokens, dtype=torch.long)
        except BaseException as error:
            self._fail(error)
            raise

    def dispatch_prepared_prefill(self, model, batch, prepared):
        if tuple(batch.request_ids) != prepared:
            raise ValueError("Prepared V4.1 prefill request IDs changed")
        terminal = any(start + count >= length for start, count, length in zip(
            batch.chunk_starts, batch.chunk_lens, batch.prompt_lens, strict=True))
        return self._submit(batch.request_ids, batch.chunk_lens, batch.chunk_starts,
                            device_sampling=batch.allow_device_greedy_sampling or not terminal,
                            stop_masks=batch.stop_token_masks)

    def reclaim_prepared_prefill(self, pending):
        logits, tokens = self._reclaim(pending)
        return PrefillResult(last_hidden=None, logits=logits, sampled_token_ids=tokens)

    def dispatch_prepared_decode(self, model, batch, prepared):
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("Prepared V4.1 decode request IDs changed")
        return self._submit(batch.request_ids, [1] * len(batch.request_ids),
                            [int(n) - 1 for n in batch.seq_lens.tolist()],
                            device_sampling=batch.allow_device_greedy_sampling,
                            stop_masks=batch.stop_token_masks)

    def reclaim_prepared_decode(self, pending):
        logits, tokens = self._reclaim(pending)
        return DecodeResult(hidden_states=None, logits=logits, sampled_token_ids=tokens)

    def run_prefill(self, model, batch):
        return self.reclaim_prepared_prefill(self.dispatch_prepared_prefill(
            model, batch, self.prepare_prefill(model, batch, buffer_slot=0)))

    def run_prefill_in_slot(self, model, batch, *, buffer_slot):
        return self.run_prefill(model, batch)

    def run_decode(self, model, batch):
        return self.reclaim_prepared_decode(self.dispatch_prepared_decode(
            model, batch, self.prepare_decode(model, batch, buffer_slot=0)))

    def run_prepared_decode(self, model, batch, prepared):
        return self.reclaim_prepared_decode(self.dispatch_prepared_decode(model, batch, prepared))

    def close(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.shutdown()
