# Torch backend for serving comparisons

The optional `--backend torch` adapter reuses vLLM Ascend 0.23.0's
`NPUWorker`, `NPUModelRunner`, model loader, attention and sampler.
Serving retains its HTTP API, engine, scheduler, IPC, request lifecycle and KV
page allocator. The adapter translates scheduled token counts and block tables
into vLLM worker inputs. Block zero remains reserved in the vLLM pool.

The initial supported path is dense Qwen3, BF16 weights and KV, one A2/A3 NPU,
and a resident full-attention cache. vLLM and vLLM Ascend are optional dependencies
required at 0.23.0 for this adapter. Use the corresponding TorchNPU/CANN stack.

`--torch-graph-mode NONE` selects eager execution. FULL and PIECEWISE use the
normal vLLM Ascend compilation and graph-capture path. For a fair comparison,
configure native vLLM with the same graph mode and scheduling flag
(`--async-scheduling` on both sides, or `--no-async-scheduling` on both), weights,
dtype, block size, prefix-cache setting,
token budget, request limit, sampling and request data. Serving may choose
different batch shapes or dispatch prefill and decode separately. Those choices
remain part of the scheduler comparison.

Complete prompts are registered before chunked prefill. Page growth is
append-only until an explicit restart. Finished requests release the worker's
cached state before the serving allocator can reuse pages. Ordinary sampling
uses the vLLM sampler. Requests requiring a host policy receive logits and the
committed token is reconciled before their next decode.

Validate deterministic outputs, chunked prefill, batched decode, prefix hits,
restart/release and HTTP streaming before reporting latency or throughput.
Performance logs should identify both the shared execution path and every
remaining configuration or batch-shape difference.

The default Torch path enables the asynchronous engine and model runner.
The serving worker prepares immutable metadata on its command lane, submits
prefill/decode on its device lane, and materializes the native async output on
its output lane. Device-side sampled tokens feed the next decode. Terminal
prefill sampling uses the output lane as well. Host constraints and mixed
commands retain ordered fallbacks where required by the common worker.

Requests rejoining the batch recover committed CPU token history when no prior
GPU row exists. A failed reclaim wakes recovery waiters, and release remains
behind prior output completion. Captured prompt lengths govern delayed prefill
results even when a later registration reuses the same request ID.

## DeepSeek V4.1 A3 W8A8 with vLLM 0.30

This is a separate text-only adapter for a checkpoint identified by
`deepseek_v41` and `quantization_config` containing `quant_method=ascend` and
`model_quant_type=W8A8_DYNAMIC`. It reads nested `text_config` and
`quant_model_weights.safetensors.index.json`. BF16 describes activations and
Serving metadata; the weight payload remains W8A8. This is not the V4
compressed-tensors checkpoint or the A5 FP8/FP4 path.

Use an isolated vLLM/vLLM Ascend 0.30 environment with the matching A3
TorchNPU/CANN stack. Qwen continues to target 0.23; this extension does not
upgrade it. The native stack must actually register `DeepseekV41ForCausalLM`.

Serving owns admission, request scheduling, logical capacity, IPC and HTTP.
The adapter sends `SchedulerOutput` to the native multiprocessing executor,
without using its Scheduler, LLMEngine or AsyncLLM. Native device sampling
feeds the next decode; CPU output reclamation stays on the output lane.
Full prompts are registered before chunking, batch re-entry recovers committed
CPU history, and release/restart frees native request state.

V4.1 has aliased SWA, compressed MLA and recurrent-state resources in one
physical block pool. The adapter also reuses the native hybrid KV manager
instead of treating its groups as independent Qwen page tables. Serving uses
a conservative logical page budget; `--total-kv-pages` caps that logical
budget, not vLLM's physical count. Startup logs report both capacities.
For comparisons, record this boundary and align native physical KV capacity
using the logged count. This does not compare two independent physical KV
allocators.

Initial scope is one serving replica, a native tensor-parallel group with TP
equal to the selected device count, and DP=EP=1. MoE uses native TP sharding.
Prefix reuse, expert/data parallel dispatch, images, speculative decoding,
PD/offload and host-logit structured constraints are excluded. Prefix caching
must be disabled because recurrent/Engram state restoration is not implemented.

These commands are templates. A real checkpoint and sufficient free devices
are required to validate model loading, generation and streaming:

```bash
pypto-serving --model /path/to/DeepSeek-V4.1-Flash-w8a8 \
  --backend torch --platform a2a3 --devices 0,1,2,3,4,5,6,7 --tp 8 --dp 1 --ep 1 \
  --dtype bfloat16 --block-size 128 --max-model-len 8192 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --long-prefill-token-threshold 512 \
  --no-enable-prefix-caching --torch-graph-mode NONE --async-scheduling \
  --served-model-name deepseek-v41 --port 8000
```

The native baseline uses the same quantization, TP, context, batching,
prefix-cache, sampling and scheduling settings:

```bash
vllm serve /path/to/DeepSeek-V4.1-Flash-w8a8 --quantization ascend \
  --dtype bfloat16 --tensor-parallel-size 8 --block-size 128 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 512 --no-enable-prefix-caching \
  --enforce-eager --async-scheduling --served-model-name deepseek-v41 --port 8001
```

Host tests can run without payloads and cover checkpoint identity, hybrid
group tables, async submission, output reordering, recovery, failure propagation
and release. They do not establish accuracy or device compatibility. The full
W8A8 model has not yet passed device/HTTP acceptance; do not report performance
from this path until that validation succeeds.
