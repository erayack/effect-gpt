# Effect GPT

<p align="center">
  <a href="https://www.npmjs.com/package/effect-gpt"><img src="https://img.shields.io/npm/v/effect-gpt.svg?style=flat-square" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/effect-gpt"><img src="https://img.shields.io/npm/dm/effect-gpt.svg?style=flat-square" alt="npm downloads"></a>
  <a href="https://github.com/erayack/RustGPT/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license"></a>
</p>

A transformer-based LLM built from scratch with [Effect](https://effect.website). Inspired by [RustGPT](https://github.com/tekaratzas/RustGPT).

## Installation

```bash
npm i effect-gpt
```

## What This Is

A complete LLM implementation including:
- **Tokenization** — BPE-style text preprocessing
- **Transformer Architecture** — embeddings, multi-head attention, feed-forward layers, layer norm
- **Training** — cross-entropy loss, backpropagation, Adam optimizer with gradient clipping
- **Inference** — greedy decoding for text generation

## Why Effect?

This project leverages Effect's robust ecosystem to bring systems-programming discipline to TypeScript:

- **Service-Based Architecture** — Uses `Context` and `Layer` to keep the core model pure and make testing deterministic by swapping implementations.
- **Type-Safe Errors** — Implements `Data.TaggedError` for a precise, union-based error system, ensuring all failure cases are handled explicitly.
- **Lazy Streaming** — Utilizes `Stream` for high-performance, backpressured data loading and batching during training.
- **Resource Management** — Uses `Scope` to guarantee that file handles and fibers are always cleaned up correctly.
- **Declarative Concurrency** — Leverages Effect's runtime to manage parallel preprocessing and training loops.

## Deep Dive

Check out our blog post: [Building a transformer-based LLM with Effect](https://hackmd.io/jW2kapAMSkWpG_PW1o3hAA).

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
```

## Quick Start

```typescript
import { Effect, Layer } from "effect"
import {
  LLM,
  Vocab,
  tokenize,
  train,
  makeLLMLayer,
  makeTrainingConfigLayer,
  SeededRandomLive,
  SeedLayer,
  ConsoleLoggerLive,
  NoOpMetricsLive
} from "effect-gpt"

// Build a tiny model
const vocab = Vocab.fromCorpus("hello world", 50)
const config = { vocabSize: vocab.size, embedDim: 32, numHeads: 2, numLayers: 2, seqLen: 16 }

const program = Effect.gen(function* () {
  const llm = yield* LLM.init(config)
  const tokens = tokenize(vocab, "hello")
  const output = yield* llm.forward(tokens)
  console.log("Output shape:", output.rows, "x", output.cols)
})

const live = Layer.mergeAll(
  SeedLayer(42),
  SeededRandomLive,
  ConsoleLoggerLive,
  NoOpMetricsLive
)

Effect.runPromise(program.pipe(Effect.provide(live)))
```

## Examples

### Tensor Operations

```typescript
import { T2D, TensorOps, seeded } from "effect-gpt"

const rng = seeded(123)

// Create tensors
const a = T2D.randn(rng, 3, 4)
const b = T2D.randn(rng, 4, 2)

// Matrix multiplication
const c = TensorOps.matmul(a, b) // 3x2

// Element-wise operations
const scaled = TensorOps.scale(a, 0.5)
const added = TensorOps.add(a, a)

// Softmax (row-wise)
const probs = TensorOps.softmaxRows(a)
```

### Training a Model

```typescript
import { Effect, Stream, Layer } from "effect"
import {
  LLM,
  Vocab,
  Dataset,
  trainStream,
  Adam,
  makeLLMLayer,
  makeTrainingConfigLayer,
  SeededRandomLive,
  SeedLayer,
  ConsoleLoggerLive,
  InMemoryMetricsLive
} from "effect-gpt"

const trainingProgram = Effect.gen(function* () {
  const vocab = Vocab.fromCorpus("the quick brown fox jumps over the lazy dog", 100)
  const config = { vocabSize: vocab.size, embedDim: 64, numHeads: 4, numLayers: 2, seqLen: 32 }

  const llm = yield* LLM.init(config)
  const optimizer = Adam.create(0.001)

  // Stream-based training with backpressure
  const batches = Stream.fromIterable([
    { input: [1, 2, 3, 4], target: [2, 3, 4, 5] },
    { input: [5, 6, 7, 8], target: [6, 7, 8, 9] }
  ])

  yield* trainStream(llm, optimizer, batches, { epochs: 10 })
})

const live = Layer.mergeAll(
  SeedLayer(42),
  SeededRandomLive,
  ConsoleLoggerLive,
  InMemoryMetricsLive
)

Effect.runPromise(trainingProgram.pipe(Effect.provide(live)))
```

### Custom Services

```typescript
import { Effect, Layer } from "effect"
import { Logger, Random, Metrics, log, next, counter } from "effect-gpt"

// Use built-in services
const program = Effect.gen(function* () {
  yield* log("info", "Starting training...")
  
  const rand = yield* next() // random float [0, 1)
  
  const trainCounter = yield* counter("batches_processed")
  trainCounter.inc()
})

// Swap implementations for testing
const testLayer = Layer.mergeAll(
  NullLoggerLive,      // silent logging
  SeededRandomLive,    // deterministic RNG
  InMemoryMetricsLive  // capture metrics
)
```


## License

MIT
