# Rust Parity Testing Plan (TypeScript)

This document outlines steps to ensure deterministic, architecture-parity testing between the Rust and TypeScript implementations of the model.

## Tiered Test Fixture Implementation

### Tier 0 — Runner & Config
- Add `vitest` + `@types/node`; scripts `test`, `test:watch`, `test:unit` in `package.json`.
- Create `tsconfig.test.json` extending `tsconfig.json` with `types: ["vitest/globals"]`.
- Location: repo root; no code changes yet.

### Tier 1 — Deterministic Randomness
- Add `src/tensor/random.ts` (`Rng`, `seeded(seed: number)`; mulberry32/PCG).
- Change `ops.ts:initNormal` to accept optional `Rng`; default to `Math.random`.
- Plumb optional RNG through constructors of `SelfAttention`, `FeedForward`, `Embeddings`, `OutputProjection`, `LayerNorm`.
- Location: `src/tensor/ops.ts`, `src/model/*.ts` (weight inits), new `src/tensor/random.ts`.

### Tier 2 — Test Utilities
- Add `tests/ts/support/tensorMatchers.ts` (`expectShape`, `expectClose` with epsilon default `1e-5`).
- Add `tests/ts/support/runEffect.ts` to run `Effect` programs inside vitest (`Effect.runPromise`).
- Add `tests/ts/support/seed.ts` helper to build seeded RNG for tests.
- Location: `tests/ts/support/*`.

### Tier 3 — Unit Parity Tests
- `self_attention.test.ts`: shape across seq lens; backward shape + weight update.
- `transformer_block.test.ts`: shape; zero-init path should act as residual.
- `layer_norm.test.ts`: per-row mean≈0/var≈1; gamma/beta grads update.
- `feed_forward.test.ts`, `embeddings.test.ts`, `output_projection.test.ts`: shape + parameter-count checks and weight update on backward.
- Location: `tests/ts/*.test.ts`.

### Tier 4 — Integration Tests (LLM)
- `llm_tokenize.test.ts`: encode/decode, ensures `</s>` appended.
- `llm_predict.test.ts`: stub OutputProjection forces EOS after N loops; compare decoded output.
- `llm_parameters.test.ts`: total parameter count matches Rust formula (`EMBEDDING_DIM`, `HIDDEN_DIM`, `MAX_SEQ_LEN`).
- `train_loop.test.ts`: tiny corpus, 1–2 epochs, seeded RNG; assert loss down or weights mutate.
- Location: `tests/ts/*.test.ts` (integration subset).

### Tier 5 — Golden Fixtures (Optional but High Value)
- Script (`dump-rust-fixtures.rs` or reuse Rust binary) to emit deterministic forward outputs; store under `tests/fixtures/rust/*.json`.
- TS tests load fixtures for early layers (embeddings, attention, transformer block) and assert outputs `expectClose` within tolerance.
- Document canonical seed (e.g., `1337`) and tolerance in `tests/README.md`.
- Location: `tests/fixtures/rust/*`, `tests/ts/*.test.ts`, `tests/README.md`.

### Tier 6 — Side Constraints & Hygiene
- Keep IO behind Effect; tests use pure constructors + in-memory data.
- Avoid `bigint`/non-ASCII in fixtures.
- Standard tolerance: `1e-5` unless otherwise noted; prefer property assertions over deep snapshots beyond early layers.

### Files to Update/Add (rolled-up)
- Root: `package.json`, `tsconfig.test.json`.
- Tensor/Model: `src/tensor/ops.ts`, `src/tensor/random.ts`, `src/model/{SelfAttention,FeedForward,Embeddings,OutputProjection,LayerNorm}.ts` (RNG injection).
- Tests: `tests/ts/support/*`, `tests/ts/*.test.ts`, optional `tests/fixtures/rust/*`, `tests/README.md`.
