# Effect GPT

A transformer-based LLM built from scratch with Effect. Inspired by [RustGPT](https://github.com/tekaratzas/RustGPT).

## What This Is

A complete LLM implementation including:
- **Tokenization** — BPE-style text preprocessing
- **Transformer Architecture** — embeddings, multi-head attention, feed-forward layers, layer norm
- **Training** — cross-entropy loss, backpropagation, Adam optimizer with gradient clipping
- **Inference** — greedy decoding for text generation

## Quick Start

```bash
bun install
bun run dev      # train + generate
bun test         # run test suite
```

## Why Effect?

This project leverages Effect's robust ecosystem to bring systems-programming discipline to TypeScript:

- **Service-Based Architecture** — Uses `Context` and `Layer` to keep the core model pure and make testing deterministic by swapping implementations (e.g., swapping a terminal logger for a test capture).
- **Type-Safe Errors** — Implements `Data.TaggedError` for a precise, union-based error system, ensuring all failure cases (shape mismatches, IO errors) are handled explicitly.
- **Lazy Streaming** — Utilizes `Stream` for high-performance, backpressured data loading and batching during training.
- **Resource Management** — Uses `Scope` to guarantee that file handles and fibers are always cleaned up correctly.
- **Declarative Concurrency** — Leverages Effect's runtime to manage parallel preprocessing and training loops without the complexity of manual orchestration.

## Deep Dive

Curious about how this was built? Check out our blog post: [Building a transformer-based LLM with Effect](https://hackmd.io/jW2kapAMSkWpG_PW1o3hAA).

## Project Structure

```
src/
├── tensor/      # Tensor2D, matmul, softmax, layer norm
├── model/       # embeddings, attention, transformer blocks, forward/backward
├── training/    # loss, gradients, Adam optimizer, training loop
├── tokenize/    # text → tokens
├── vocab/       # vocabulary management
├── services/    # Effect services (Random, Logger, Metrics)
└── cli/         # command-line interface
tests/
├── ts/                  # TypeScript test suite
│   ├── support/         # test helpers, seeded RNG, 
│   └── ...              # model, tensor, and training unit/integration tests
```
