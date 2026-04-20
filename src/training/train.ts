import * as Effect from "effect/Effect"
import * as Stream from "effect/Stream"
import * as Chunk from "effect/Chunk"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"
import * as Ref from "effect/Ref"
import * as Option from "effect/Option"
import type { ShapeError } from "../tensor/ops"
import * as Ops from "../tensor/ops"
import * as T from "../tensor/Tensor2D"
import { tokenize } from "../tokenize/tokenize"
import type { LLM } from "../model/LLM"
import { runBackwardPass, runForwardPass, type SequenceLayout } from "../model/ModelLayer"
import { crossEntropyLossAndDLogitsFromLogits } from "./loss"
import { clipGlobalL2 } from "./clip"
import type { LoggerServiceId } from "../services/Logger"
import { info } from "../services/Logger"
import type { MetricsServiceId } from "../services/Metrics"
import { counter, gauge, timed } from "../services/Metrics"
import { TrainingError } from "../errors"
import type { TrainingError as TrainingErrorType } from "../errors"

export interface TrainingConfig {
  readonly epochs: number
  readonly learningRate: number
  readonly clipNorm?: number
  readonly trainConcurrency?: number
}

export interface PreprocessSettings {
  readonly concurrency: number | "unbounded"
  readonly batchSize: number
  readonly cacheScope?: "perRun" | "perEpoch"
}

class TrainingConfigTag extends Context.Tag("effect-gpt/training/TrainingConfig")<TrainingConfigTag, TrainingConfig>() {}

class LLMServiceTag extends Context.Tag("effect-gpt/training/LLMService")<LLMServiceTag, LLM>() {}

class PreprocessSettingsTag extends Context.Tag("effect-gpt/training/PreprocessSettings")<
  PreprocessSettingsTag,
  PreprocessSettings
>() {}

export const TrainingConfig = TrainingConfigTag
export const LLMService = LLMServiceTag
export const PreprocessSettings = PreprocessSettingsTag

export const makeLLMLayer = (llm: LLM) => Layer.succeed(LLMService, llm)
export const makeTrainingConfigLayer = (config: TrainingConfig) =>
  Layer.succeed(TrainingConfig, config)
export const makePreprocessSettingsLayer = (settings: PreprocessSettings) =>
  Layer.succeed(PreprocessSettings, settings)
export const DefaultPreprocessSettings: PreprocessSettings = Object.freeze({
  concurrency: "unbounded",
  batchSize: 1,
  cacheScope: "perEpoch"
})
export const CachedPreprocessSettings: PreprocessSettings = Object.freeze({
  ...DefaultPreprocessSettings,
  cacheScope: "perRun"
})
export const DefaultPreprocessSettingsLive = makePreprocessSettingsLayer(DefaultPreprocessSettings)
export const CachedPreprocessSettingsLive = makePreprocessSettingsLayer(CachedPreprocessSettings)

type TrainEnv =
  | TrainingConfigTag
  | LLMServiceTag
  | LoggerServiceId
  | MetricsServiceId
  | PreprocessSettingsTag

const mapShapeError = <A, R>(effect: Effect.Effect<A, ShapeError, R>) =>
  effect.pipe(Effect.mapError(TrainingError.shape))

const mapShapeUnknown = (error: unknown): TrainingErrorType =>
  error instanceof Ops.ShapeError ? TrainingError.shape(error) : TrainingError.fromUnknown(error)

const wrapThrowing = <A>(
  thunk: () => A,
  mapError: (error: unknown) => TrainingErrorType = TrainingError.fromUnknown
) =>
  Effect.try({
    try: thunk,
    catch: (error) => mapError(error)
  })

const resolveTrainConcurrency = (
  value: number | undefined
): Effect.Effect<number, TrainingErrorType> =>
  Effect.gen(function* () {
    if (value === undefined) {
      return 1
    }
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
      return yield* TrainingError.config("trainConcurrency must be a positive integer")
    }
    if (value > 1) {
      return yield* TrainingError.config(
        "trainConcurrency > 1 is not supported for batched training because model updates are applied sequentially on shared mutable state"
      )
    }
    return value
  })

interface TrainingExample {
  readonly inputIds: ReadonlyArray<number>
  readonly targetIds: ReadonlyArray<number>
}

interface TrainingBatch {
  readonly inputIds: Int32Array
  readonly targetIds: Int32Array
  readonly exampleCount: number
  readonly tokenCount: number
  readonly layout: SequenceLayout
}

interface PreparedTrainingCorpus {
  readonly batches: ReadonlyArray<TrainingBatch>
  readonly totalExamples: number
  readonly totalTokens: number
}

const buildTrainingBatch = (examples: ReadonlyArray<TrainingExample>): TrainingBatch => {
  const exampleCount = examples.length
  let tokenCount = 0
  for (const example of examples) {
    tokenCount += example.inputIds.length
  }

  const inputIds = new Int32Array(tokenCount)
  const targetIds = new Int32Array(tokenCount)
  const sequenceIds = new Int32Array(tokenCount)
  const positionIds = new Int32Array(tokenCount)
  const sequenceLengths = new Array<number>(exampleCount)

  let offset = 0
  for (let batchIndex = 0; batchIndex < exampleCount; batchIndex++) {
    const example = examples[batchIndex]!
    const seqLen = example.inputIds.length
    sequenceLengths[batchIndex] = seqLen
    for (let i = 0; i < seqLen; i++) {
      const tokenIndex = offset + i
      inputIds[tokenIndex] = example.inputIds[i]!
      targetIds[tokenIndex] = example.targetIds[i]!
      sequenceIds[tokenIndex] = batchIndex
      positionIds[tokenIndex] = i
    }
    offset += seqLen
  }

  return {
    inputIds,
    targetIds,
    exampleCount,
    tokenCount,
    layout: {
      totalTokens: tokenCount,
      sequenceLengths,
      sequenceIds,
      positionIds
    }
  }
}

const prepareTrainingCorpus = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>,
  preprocess: (text: string) => Effect.Effect<Option.Option<TrainingExample>, TrainingErrorType>,
  settings: PreprocessSettings
): Effect.Effect<PreparedTrainingCorpus, TrainingErrorType, R> =>
  Effect.gen(function* () {
    const batchSize = Math.max(1, settings.batchSize)
    const preprocessed = makeStream().pipe(
      Stream.mapError(TrainingError.fromUnknown),
      Stream.mapEffect(preprocess, { concurrency: settings.concurrency }),
      Stream.filterMap((value) => value)
    )

    const examples = yield* Stream.runCollect(preprocessed)
    const batches = Array.from(
      Chunk.chunksOf(batchSize)(examples),
      (chunk) => buildTrainingBatch(Array.from(chunk))
    )

    let totalExamples = 0
    let totalTokens = 0
    for (const batch of batches) {
      totalExamples += batch.exampleCount
      totalTokens += batch.tokenCount
    }

    return {
      batches,
      totalExamples,
      totalTokens
    }
  })

const trainWithStreamFactory = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>,
  defaultCacheScope: "perRun" | "perEpoch"
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> =>
  Effect.gen(function* () {
    const llm = yield* LLMService
    const config = yield* TrainingConfig
    const preprocessSettings = yield* PreprocessSettings

    const clipNorm = config.clipNorm ?? 5.0
    const cacheScope = preprocessSettings.cacheScope ?? defaultCacheScope
    const trainConcurrency = yield* resolveTrainConcurrency(config.trainConcurrency)

    const epochCounter = yield* counter("epochs_completed")
    const lossGauge = yield* gauge("epoch_loss")
    const examplesCounter = yield* counter("examples_processed")

    const preprocess = Effect.fn("Train.preprocess")(function* (text: string) {
      const tokens = [...tokenize(text, llm.vocab)]
      if (tokens.length < 2) {
        return Option.none<TrainingExample>()
      }

      return Option.some({
        inputIds: tokens.slice(0, tokens.length - 1),
        targetIds: tokens.slice(1)
      } satisfies TrainingExample)
    })

    const runEpoch = Effect.fn("Train.runEpoch")(function* (
      epoch: number,
      corpus: PreparedTrainingCorpus
    ) {
      const epochResult = yield* timed(`epoch_${epoch}`, Effect.gen(function* () {
        const totalLossRef = yield* Ref.make(0)

        const trainBatch = Effect.fn("Train.trainBatch")(function* (batch: TrainingBatch) {
          let input = T.fromArray(1, batch.tokenCount, batch.inputIds)
          const cacheKey = Symbol("train-batch")
          const context = { sequenceLayout: batch.layout, cacheKey, captureCache: true }
          input = yield* mapShapeError(runForwardPass(llm.network, input, context))

          const logits = input
          const { loss, grads: initialGrads } = yield* wrapThrowing(
            () => crossEntropyLossAndDLogitsFromLogits(logits, batch.targetIds),
            mapShapeUnknown
          )
          yield* Ref.update(totalLossRef, (current) => current + loss * batch.tokenCount)

          let grads = initialGrads
          clipGlobalL2(grads, clipNorm)

          grads = yield* mapShapeError(runBackwardPass(llm.network, grads, config.learningRate, cacheKey))
        })

        yield* Effect.forEach(corpus.batches, trainBatch, { concurrency: trainConcurrency, discard: true })

        const totalLoss = yield* Ref.get(totalLossRef)
        const totalTokens = corpus.totalTokens
        const totalExamples = corpus.totalExamples
        yield* examplesCounter.inc(totalExamples)
        return { totalLoss, totalTokens, totalExamples }
      }).pipe(Effect.withSpan("Train.epochLoop")))

      const { totalLoss, totalTokens, totalExamples } = epochResult.value
      const avgLoss = totalTokens > 0 ? totalLoss / totalTokens : 0

      yield* lossGauge.set(avgLoss)
      yield* epochCounter.inc()
      yield* info(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`, {
        epoch,
        loss: avgLoss,
        examples: totalExamples,
        durationMs: epochResult.durationMs
      })
    })

    if (config.epochs <= 0) {
      return
    }

    const prepareEpochCorpus = () => prepareTrainingCorpus(makeStream, preprocess, preprocessSettings)

    if (cacheScope === "perRun") {
      const corpus = yield* prepareEpochCorpus()
      for (let epoch = 0; epoch < config.epochs; epoch++) {
        yield* runEpoch(epoch, corpus)
      }
      return
    }

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      const corpus = yield* prepareEpochCorpus()
      yield* runEpoch(epoch, corpus)
    }
  }).pipe(Effect.withSpan("Train.trainWithStreamFactory"))

export const train = (
  examples: ReadonlyArray<string>
): Effect.Effect<void, TrainingErrorType, TrainEnv> =>
  trainWithStreamFactory(() => Stream.fromIterable(examples), "perRun")

export const trainStream = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> => trainWithStreamFactory(makeStream, "perEpoch")
