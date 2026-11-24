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
