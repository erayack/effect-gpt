# Performance Tracker

This file tracks the current status of the performance findings against the codebase as it exists now.

## Constraint

Performance work in this repo should preserve the Effect-first architecture.

- Keep `Effect` in orchestration, streaming, dependency injection, logging, metrics, training control flow, and public model-layer APIs.
- Optimize hot internals without abandoning `Effect` as the architectural model.
- Favor direct typed-array loops, smaller caches, fewer allocations, and less recomputation inside kernels and backward passes over removing `Effect` from the surrounding design.

## Done

- [x] **Hot layers now reuse grow-only workspaces and output buffers**
  - Implemented in:
    - [src/tensor/Workspace.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/Workspace.ts:1)
    - [src/model/Embeddings.ts](/Users/erayack/Desktop/code/RustGPT/src/model/Embeddings.ts:1)
    - [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1)
    - [src/model/FeedForward.ts](/Users/erayack/Desktop/code/RustGPT/src/model/FeedForward.ts:1)
    - [src/model/LayerNorm.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LayerNorm.ts:1)
    - [src/model/OutputProjection.ts](/Users/erayack/Desktop/code/RustGPT/src/model/OutputProjection.ts:1)
    - [src/model/TransformerBlock.ts](/Users/erayack/Desktop/code/RustGPT/src/model/TransformerBlock.ts:1)
    - [src/model/LLM.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LLM.ts:1)
  - `TensorWorkspace.borrowTensor()` now reuses grow-only backing buffers instead of exact-shape allocations, cache-bearing layer forwards retain pooled workspaces until the matching backward releases them, and incremental decode/prefill paths reuse per-generation scratch state instead of allocating fresh tensor/vector buffers every token.

- [x] **Structural: Reduce Effect/runtime overhead inside math-heavy paths**
  - Implemented in:
    - [src/model/ModelLayer.ts](/Users/erayack/Desktop/code/RustGPT/src/model/ModelLayer.ts:1)
    - [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:1)
    - [src/model/Embeddings.ts](/Users/erayack/Desktop/code/RustGPT/src/model/Embeddings.ts:1)
    - [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1)
    - [src/model/FeedForward.ts](/Users/erayack/Desktop/code/RustGPT/src/model/FeedForward.ts:1)
    - [src/model/LayerNorm.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LayerNorm.ts:1)
    - [src/model/OutputProjection.ts](/Users/erayack/Desktop/code/RustGPT/src/model/OutputProjection.ts:1)
    - [src/model/TransformerBlock.ts](/Users/erayack/Desktop/code/RustGPT/src/model/TransformerBlock.ts:1)
    - [src/model/LLM.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LLM.ts:1)
    - [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:1)
  - Hot tensor kernels now expose synchronous internal variants, built-in layers run their math-heavy forward/backward paths synchronously behind the existing `Effect` API, and training/inference traverse sync-capable networks inside a small number of outer `Effect` boundaries. Public orchestration and service integration remain `Effect`-first, while mixed custom networks still fall back to the effect-per-layer path.

- [x] **Structural: Replace naïve JS GEMM with a stronger backend**
  - Implemented in:
    - [src/tensor/gemm.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/gemm.ts:1)
    - [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:1)
    - [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1)
    - [src/model/FeedForward.ts](/Users/erayack/Desktop/code/RustGPT/src/model/FeedForward.ts:1)
    - [src/model/OutputProjection.ts](/Users/erayack/Desktop/code/RustGPT/src/model/OutputProjection.ts:1)
  - `Ops.matMul()` / `matMulInto()` now route through a transpose-aware blocked JS GEMM backend with caller-owned scratch reuse. Backward and attention paths now use transpose flags instead of materializing temporary transposed tensors for GEMM-only use.

- [x] **Fuse attention Q/K/V projection storage and projection GEMMs**
  - Implemented in [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1).
  - `SelfAttention` now stores Q/K/V projection weights in one fused `wQKV` tensor, projects all three activations with one GEMM, and collapses the projection-side backward pass to one fused weight-gradient GEMM plus one fused input-gradient GEMM.
  - The fused storage is laid out as `[embeddingDim, 3 * embeddingDim]` so each unpacked Q/K/V block matches the original `input * wQ`, `input * wK`, and `input * wV` math exactly.

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

- [x] **Medium: Reuse scratch buffers / add lower-allocation tensor kernels**
  - Implemented in:
    - [src/tensor/Workspace.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/Workspace.ts:1)
    - [src/tensor/ops.ts](/Users/erayack/Desktop/code/RustGPT/src/tensor/ops.ts:16)
    - [src/model/Embeddings.ts](/Users/erayack/Desktop/code/RustGPT/src/model/Embeddings.ts:1)
    - [src/model/FeedForward.ts](/Users/erayack/Desktop/code/RustGPT/src/model/FeedForward.ts:1)
    - [src/model/LayerNorm.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LayerNorm.ts:1)
    - [src/model/OutputProjection.ts](/Users/erayack/Desktop/code/RustGPT/src/model/OutputProjection.ts:1)
    - [src/model/SelfAttention.ts](/Users/erayack/Desktop/code/RustGPT/src/model/SelfAttention.ts:1)
    - [src/training/loss.ts](/Users/erayack/Desktop/code/RustGPT/src/training/loss.ts:1)
    - [src/training/train.ts](/Users/erayack/Desktop/code/RustGPT/src/training/train.ts:1)
    - [src/model/LLM.ts](/Users/erayack/Desktop/code/RustGPT/src/model/LLM.ts:1)
  - Added caller-supplied-buffer and in-place tensor kernels, introduced method-local scratch workspaces for ephemeral intermediates, removed the training softmax allocation by computing loss and gradients directly from logits, and switched token selection back to direct argmax over logits. Returned tensors and cached backward state remain caller-owned so the Effect-first public architecture is unchanged.

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
