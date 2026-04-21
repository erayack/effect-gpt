import { describe, test, expect } from "bun:test"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Stream from "effect/Stream"
import { makeLLM } from "./support/factories"
import { expectNotClose } from "./support/tensorMatchers"
import { CANONICAL_SEED } from "./support/seed"
import {
  train,
  trainStream,
  makeLLMLayer,
  makeTrainingConfigLayer,
  makePreprocessSettingsLayer
} from "../../src/training/train"
import { Embeddings } from "../../src/model/Embeddings"
import { TransformerBlock } from "../../src/model/TransformerBlock"
import { OutputProjection } from "../../src/model/OutputProjection"
import * as T from "../../src/tensor/Tensor2D"
import { SilentLoggerLive, Logger } from "../../src/services/Logger"
import { NoOpMetricsLive } from "../../src/services/Metrics"
import { TestServicesLayer as BaseTestServicesLayer } from "./support/stubs"

const TestServicesLayer = Layer.mergeAll(
  BaseTestServicesLayer,
  makePreprocessSettingsLayer({ concurrency: "unbounded", batchSize: 1 })
)

const BatchedTestServicesLayer = Layer.mergeAll(
  BaseTestServicesLayer,
  makePreprocessSettingsLayer({ concurrency: "unbounded", batchSize: 2 })
)

describe("Train Loop", () => {
  const tinyVocab = ["hello", "world", "is", "this", "test", "</s>"]
  const tinyCorpus = ["hello world </s>", "this is </s>", "test world </s>"]

  const createTinyLLM = (seed: number = CANONICAL_SEED) =>
    makeLLM({ seed, vocabWords: tinyVocab, numTransformerBlocks: 1 })

  test("training mutates embeddings weights", () => {
    const llm = createTinyLLM()
    const embeddings = llm.network[0] as Embeddings
    const tokenEmbeddingsBefore = T.clone(embeddings.tokenEmbeddings)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )

    expectNotClose(embeddings.tokenEmbeddings, tokenEmbeddingsBefore)
  })

  test("training mutates transformer weights", () => {
    const llm = createTinyLLM()
    const transformer = llm.network[1] as TransformerBlock
    const w1Before = T.clone(transformer.feedForward.w1)
    const wQKVBefore = T.clone(transformer.attention.wQKV)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )

    expectNotClose(transformer.feedForward.w1, w1Before)
    expectNotClose(transformer.attention.wQKV, wQKVBefore)
  })

  test("training mutates output projection weights", () => {
    const llm = createTinyLLM()
    const output = llm.network[2] as OutputProjection
    const wOutBefore = T.clone(output.wOut)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )

    expectNotClose(output.wOut, wOutBefore)
  })

  test("loss decreases over epochs", async () => {
    const llm = createTinyLLM()
    const losses: Array<number> = []

    const makeCaptureLossLogger = () => {
      const service = {
        log: (_level: any, _message: string, data?: Record<string, unknown>) => {
          if (data?.loss !== undefined) {
            losses.push(data.loss as number)
          }
          return Effect.void
        },
        debug: () => Effect.void,
        info: (_message: string, data?: Record<string, unknown>) => {
          if (data?.loss !== undefined) {
            losses.push(data.loss as number)
          }
          return Effect.void
        },
        warn: () => Effect.void,
        error: () => Effect.void
      }
      return Layer.succeed(Logger, service)
    }

    const program = train(tinyCorpus).pipe(
      Effect.provide(makeLLMLayer(llm)),
      Effect.provide(makeTrainingConfigLayer({ epochs: 3, learningRate: 0.01 })),
      Effect.provide(makeCaptureLossLogger()),
      Effect.provide(NoOpMetricsLive),
      Effect.provide(makePreprocessSettingsLayer({ concurrency: "unbounded", batchSize: 1 }))
    )

    await Effect.runPromise(program)

    expect(losses.length).toBe(3)
    expect(losses[2]).toBeLessThan(losses[0]!)
  })

  test("seeded RNG produces deterministic training", () => {
    const llm1 = createTinyLLM(CANONICAL_SEED)
    const llm2 = createTinyLLM(CANONICAL_SEED)

    const runOnce = (llm: ReturnType<typeof createTinyLLM>) =>
      Effect.runSync(
        train(tinyCorpus).pipe(
          Effect.provide(makeLLMLayer(llm)),
          Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
          Effect.provide(TestServicesLayer)
        )
      )
    runOnce(llm1)
    runOnce(llm2)

    const embeddings1 = llm1.network[0] as Embeddings
    const embeddings2 = llm2.network[0] as Embeddings

    for (let i = 0; i < embeddings1.tokenEmbeddings.data.length; i++) {
      expect(embeddings1.tokenEmbeddings.data[i]).toBe(embeddings2.tokenEmbeddings.data[i])
    }
  })

  test("empty corpus does not crash", () => {
    const llm = createTinyLLM()
    expect(() =>
      Effect.runSync(
        train([]).pipe(
          Effect.provide(makeLLMLayer(llm)),
          Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
          Effect.provide(TestServicesLayer)
        )
      )
    ).not.toThrow()
  })

  test("single example corpus trains", () => {
    const llm = createTinyLLM()
    const embeddings = llm.network[0] as Embeddings
    const before = T.clone(embeddings.tokenEmbeddings)

    Effect.runSync(
      train(["hello world </s>"]).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )

    expectNotClose(embeddings.tokenEmbeddings, before)
  })

  test("multiple epochs further mutate weights", () => {
    const llm = createTinyLLM()
    const embeddings = llm.network[0] as Embeddings

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )
    const afterEpoch1 = T.clone(embeddings.tokenEmbeddings)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )
    expectNotClose(embeddings.tokenEmbeddings, afterEpoch1)
  })

  test("training with higher learning rate causes larger weight changes", () => {
    const llm1 = createTinyLLM(CANONICAL_SEED)
    const llm2 = createTinyLLM(CANONICAL_SEED)

    const embeddings1 = llm1.network[0] as Embeddings
    const embeddings2 = llm2.network[0] as Embeddings
    const initial = T.clone(embeddings1.tokenEmbeddings)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm1)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.001 })),
        Effect.provide(TestServicesLayer)
      )
    )
    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm2)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.1 })),
        Effect.provide(TestServicesLayer)
      )
    )

    let diff1 = 0
    let diff2 = 0
    for (let i = 0; i < initial.data.length; i++) {
      diff1 += Math.abs(embeddings1.tokenEmbeddings.data[i]! - initial.data[i]!)
      diff2 += Math.abs(embeddings2.tokenEmbeddings.data[i]! - initial.data[i]!)
    }

    expect(diff2).toBeGreaterThan(diff1)
  })

  test("batchSize > 1 performs one optimizer step per minibatch", () => {
    const llm = createTinyLLM()
    const embeddings = llm.network[0] as Embeddings

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(BatchedTestServicesLayer)
      )
    )

    expect(embeddings.tokenOptimizer.timestep).toBe(2)
    expect(embeddings.positionalOptimizer.timestep).toBe(2)
  })

  test("batchSize > 1 still mutates weights", () => {
    const llm = createTinyLLM()
    const embeddings = llm.network[0] as Embeddings
    const before = T.clone(embeddings.tokenEmbeddings)

    Effect.runSync(
      train(tinyCorpus).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01 })),
        Effect.provide(BatchedTestServicesLayer)
      )
    )

    expectNotClose(embeddings.tokenEmbeddings, before)
  })

  test("trainConcurrency=1 is accepted for batched training", () => {
    const llm = createTinyLLM()

    expect(() =>
      Effect.runSync(
        train(tinyCorpus).pipe(
          Effect.provide(makeLLMLayer(llm)),
          Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01, trainConcurrency: 1 })),
          Effect.provide(BatchedTestServicesLayer)
        )
      )
    ).not.toThrow()
  })

  test("trainConcurrency > 1 is rejected for batched training", () => {
    const llm = createTinyLLM()

    expect(() =>
      Effect.runSync(
        train(tinyCorpus).pipe(
          Effect.provide(makeLLMLayer(llm)),
          Effect.provide(makeTrainingConfigLayer({ epochs: 1, learningRate: 0.01, trainConcurrency: 2 })),
          Effect.provide(BatchedTestServicesLayer)
        )
      )
    ).toThrow("trainConcurrency > 1 is not supported for batched training")
  })

  test("loss decreases over epochs with batchSize > 1", async () => {
    const llm = createTinyLLM()
    const losses: Array<number> = []

    const makeCaptureLossLogger = () => {
      const service = {
        log: (_level: any, _message: string, data?: Record<string, unknown>) => {
          if (data?.loss !== undefined) {
            losses.push(data.loss as number)
          }
          return Effect.void
        },
        debug: () => Effect.void,
        info: (_message: string, data?: Record<string, unknown>) => {
          if (data?.loss !== undefined) {
            losses.push(data.loss as number)
          }
          return Effect.void
        },
        warn: () => Effect.void,
        error: () => Effect.void
      }
      return Layer.succeed(Logger, service)
    }

    const program = train(tinyCorpus).pipe(
      Effect.provide(makeLLMLayer(llm)),
      Effect.provide(makeTrainingConfigLayer({ epochs: 3, learningRate: 0.01 })),
      Effect.provide(makeCaptureLossLogger()),
      Effect.provide(NoOpMetricsLive),
      Effect.provide(makePreprocessSettingsLayer({ concurrency: "unbounded", batchSize: 2 }))
    )

    await Effect.runPromise(program)

    expect(losses.length).toBe(3)
    expect(losses[2]).toBeLessThan(losses[0]!)
  })

  test("trainStream preserves per-epoch stream consumption by default", () => {
    const llm = createTinyLLM()
    let streamCalls = 0

    const makeStream = () => {
      streamCalls += 1
      return Stream.fromIterable(tinyCorpus)
    }

    Effect.runSync(
      trainStream(makeStream).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 3, learningRate: 0.01 })),
        Effect.provide(TestServicesLayer)
      )
    )

    expect(streamCalls).toBe(3)
  })

  test("trainStream can opt into per-run preprocessing", () => {
    const llm = createTinyLLM()
    let streamCalls = 0

    const makeStream = () => {
      streamCalls += 1
      return Stream.fromIterable(tinyCorpus)
    }

    Effect.runSync(
      trainStream(makeStream).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 3, learningRate: 0.01 })),
        Effect.provide(BaseTestServicesLayer),
        Effect.provide(
          makePreprocessSettingsLayer({
            concurrency: "unbounded",
            batchSize: 1,
            cacheScope: "perRun"
          })
        )
      )
    )

    expect(streamCalls).toBe(1)
  })

  test("epochs=0 does not consume the stream even with per-run preprocessing", () => {
    const llm = createTinyLLM()
    let streamCalls = 0

    const makeStream = () => {
      streamCalls += 1
      return Stream.fromIterable(tinyCorpus)
    }

    Effect.runSync(
      trainStream(makeStream).pipe(
        Effect.provide(makeLLMLayer(llm)),
        Effect.provide(makeTrainingConfigLayer({ epochs: 0, learningRate: 0.01 })),
        Effect.provide(BaseTestServicesLayer),
        Effect.provide(
          makePreprocessSettingsLayer({
            concurrency: "unbounded",
            batchSize: 1,
            cacheScope: "perRun"
          })
        )
      )
    )

    expect(streamCalls).toBe(0)
  })
})
