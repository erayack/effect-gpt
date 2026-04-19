# Performance Tracker

This file tracks the current status of the performance findings against the codebase as it exists now.

## Constraint

Performance work in this repo should preserve the Effect-first architecture.

- Keep `Effect` in orchestration, streaming, dependency injection, logging, metrics, training control flow, and public model-layer APIs.
- Optimize hot internals without abandoning `Effect` as the architectural model.
- Favor direct typed-array loops, smaller caches, fewer allocations, and less recomputation inside kernels and backward passes over removing `Effect` from the surrounding design.

## Remaining

### Next

- [ ] **Medium: Reuse scratch buffers / add lower-allocation tensor kernels**
  - Current state: the hot tensor ops were moved off `T.get`/`T.set`, but most ops still allocate fresh output buffers every call in [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:16).
  - Impact: allocation pressure remains high during training and inference.
  - Next step: add in-place or caller-supplied-buffer variants for common kernels where aliasing is safe.

### Later

- [ ] **Structural: Replace naïve JS GEMM with a stronger backend**
  - Current state: `Ops.matMul()` is improved but still pure JS in [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:16).
  - Impact: matrix multiply remains the main long-term throughput ceiling.
  - Next step: evaluate blocked JS, cached transposed weights, WASM, BLAS, or GPU-backed execution.

- [ ] **Structural: Reduce Effect/runtime overhead inside math-heavy paths**
  - Current state: core kernels still cross `Effect` boundaries frequently in model execution.
  - Impact: runtime overhead is smaller than the algorithmic items above, but still present in inner loops.
  - Next step: keep orchestration in `Effect`, but minimize abstraction overhead inside hot tensor/model paths.

## Done

- [x] **Pre-tokenize datasets per run / epoch**
  - Implemented in [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:1).
  - Training now supports per-run corpus preparation, reusing prepared minibatches across epochs while keeping `Effect` orchestration intact. `train()` and the CLI opt into the cached path by default, while `trainStream()` preserves per-epoch stream consumption unless callers explicitly set `cacheScope: "perRun"`.

- [x] **KV-cache based incremental decoding**
  - Implemented in:
    - [src/model/LLM.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LLM.ts:1)
    - [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1)
    - [src/model/TransformerBlock.ts](/Users/erayack/Desktop/code/RustGPT/src/model/TransformerBlock.ts:1)
  - `LLM.forward()` now prefills the prompt once, caches per-layer key/value state, and decodes subsequent tokens by projecting only the newest token while preserving the Effect-first public API. Falls back to the full-recompute path when the network does not expose the incremental interface.

- [x] **Sparse embedding gradients and sparse optimizer updates**
  - Implemented in [src/model/Embeddings.ts](/Users/erayack/Desktop/code/RustGPT/src/model/Embeddings.ts:75).
  - `Embeddings.backward()` now accumulates gradients only for the touched token and position rows and applies them via `Adam.stepRows`, so memory and update cost scale with the unique tokens in the batch instead of the full vocabulary / max position table.

- [x] **Training now performs real minibatching**
  - Implemented in [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:101).
  - The training loop now preprocesses examples, groups them into actual minibatches, flattens them with explicit sequence layout metadata, and performs one optimizer step per minibatch.

- [x] **Training concurrency behavior is explicit and safe**
  - Implemented in [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:75).
  - `trainConcurrency` now defaults to `1`, rejects invalid values, and rejects values greater than `1` because batched training mutates shared model state sequentially.

- [x] **Softmax-before-argmax was removed from inference**
  - Implemented in [src/model/LLM.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LLM.ts:101).
  - Greedy decoding now takes `argmax` directly from the final logits row.

- [x] **Training-time argmax inspection was removed**
  - Implemented in [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:176).
  - The train loop no longer computes `argmaxRows(probs)` for non-essential inspection.

- [x] **Hot tensor ops were moved off `T.get` / `T.set`**
  - Implemented in [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:16).
  - `matMul`, reductions, transpose, gather/slice, and broadcast helpers now use direct typed-array access and cached offsets.

- [x] **Attention backward now uses cached forward intermediates**
  - Implemented in [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:19).
  - `q`, `k`, `v`, and `attnWeights` are cached during forward and reused in backward.

- [x] **LayerNorm backward now uses cached forward intermediates**
  - Implemented in [src/model/LayerNorm.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LayerNorm.ts:16).
  - `normalized` and `rstd` are cached during forward and reused directly in backward instead of being recomputed.

- [x] **Clone-heavy caching was reduced in the model layers**
  - Implemented in:
    - [src/model/Embeddings.ts](/Users/erayack/Desktop/code/RustGPT/src/model/Embeddings.ts:16)
    - [src/model/OutputProjection.ts](/Users/erayack/Desktop/code/RustGPT/src/model/OutputProjection.ts:17)
    - [src/model/FeedForward.ts](/Users/erayack/Desktop/code/RustGPT/src/model/FeedForward.ts:19)
    - [src/model/LayerNorm.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LayerNorm.ts:16)
  - These layers now cache only the data required for backward and avoid unnecessary forward-time clones.
