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
import { softmaxRows, crossEntropyLossAndDLogits } from "./loss"
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
  batchSize: 1
})
export const DefaultPreprocessSettingsLive = makePreprocessSettingsLayer(DefaultPreprocessSettings)

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

const clampConcurrency = (value: number | undefined, fallback: number): number =>
  value === undefined ? fallback : Math.max(1, value)

interface TrainingExample {
  readonly inputIds: ReadonlyArray<number>
  readonly targetIds: ReadonlyArray<number>
}

const preprocessExample = (text: string, llm: LLM): Option.Option<TrainingExample> => {
  const tokens = [...tokenize(text, llm.vocab)]
  if (tokens.length < 2) {
    return Option.none()
  }

  return Option.some({
    inputIds: tokens.slice(0, tokens.length - 1),
    targetIds: tokens.slice(1)
  } satisfies TrainingExample)
}

const runTrainingExample = (
  llm: LLM,
  example: TrainingExample,
  learningRate: number,
  clipNorm: number,
  endTokenId: number
): Effect.Effect<number, TrainingErrorType> =>
  Effect.gen(function* () {
    const { inputIds, targetIds } = example

    let input = T.fromArray(1, inputIds.length, inputIds)
    for (const layer of llm.network) {
      input = yield* mapShapeError(layer.forward(input))
    }

    const logits = input
    const probs = yield* wrapThrowing(() => softmaxRows(logits), mapShapeUnknown)
    const { loss, grads: initialGrads } = yield* wrapThrowing(
      () => crossEntropyLossAndDLogits(probs, targetIds),
      mapShapeUnknown
    )

    let grads = initialGrads
    clipGlobalL2(grads, clipNorm)

    for (let i = llm.network.length - 1; i >= 0; i--) {
      grads = yield* mapShapeError(llm.network[i]!.backward(grads, learningRate))
    }

    const tokens = Ops.argmaxRows(probs)
    const nextToken = tokens[tokens.length - 1]
    if (nextToken === endTokenId) {
      return loss
    }

    return loss
  })

const trainWithStreamFactory = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> =>
  Effect.gen(function* () {
    const llm = yield* LLMService
    const config = yield* TrainingConfig
    const preprocessSettings = yield* PreprocessSettings

    const endTokenId = llm.vocab.encode("</s>")
    if (endTokenId._tag === "None") {
      return yield* TrainingError.config("End token </s> not found in vocabulary")
    }

    const clipNorm = config.clipNorm ?? 5.0
    const batchSize = Math.max(1, preprocessSettings.batchSize)
    const trainConcurrency = clampConcurrency(config.trainConcurrency, 1)

    const epochCounter = yield* counter("epochs_completed")
    const lossGauge = yield* gauge("epoch_loss")
    const examplesCounter = yield* counter("examples_processed")

    const runEpoch = Effect.fn("Train.runEpoch")(function* (epoch: number) {
      const epochResult = yield* timed(`epoch_${epoch}`, Effect.gen(function* () {
        const stream = makeStream().pipe(Stream.mapError(TrainingError.fromUnknown))
        const preprocessed = (batchSize === 1
          ? stream
          : stream.pipe(Stream.mapChunks(Chunk.chunksOf(batchSize)), Stream.flattenChunks)
        ).pipe(Stream.map((text) => preprocessExample(text, llm)), Stream.filterMap((value) => value))

        if (trainConcurrency === 1) {
          let totalLoss = 0
          let totalExamples = 0

          const trainExample = Effect.fn("Train.trainExampleSequential")((example: TrainingExample) =>
            Effect.gen(function* () {
              totalLoss += yield* runTrainingExample(
                llm,
                example,
                config.learningRate,
                clipNorm,
                endTokenId.value
              )
              totalExamples += 1
            })
          )

          yield* Stream.runForEach(preprocessed, trainExample)
          return { totalLoss, totalExamples }
        }

        const totalLossRef = yield* Ref.make(0)
        const totalExamplesRef = yield* Ref.make(0)
        const trainExample = Effect.fn("Train.trainExampleConcurrent")((example: TrainingExample) =>
          Effect.gen(function* () {
            const loss = yield* runTrainingExample(llm, example, config.learningRate, clipNorm, endTokenId.value)
            yield* Ref.update(totalLossRef, (current) => current + loss)
            yield* Ref.update(totalExamplesRef, (current) => current + 1)
          })
        )

        yield* Effect.scoped(
          Stream.runDrain(
            Stream.mapEffect(preprocessed, trainExample, { concurrency: trainConcurrency })
          )
        )

        const totalLoss = yield* Ref.get(totalLossRef)
        const totalExamples = yield* Ref.get(totalExamplesRef)
        return { totalLoss, totalExamples }
      }).pipe(Effect.withSpan("Train.epochLoop")))

      const { totalLoss, totalExamples } = epochResult.value
      const avgLoss = totalExamples > 0 ? totalLoss / totalExamples : 0

      yield* examplesCounter.inc(totalExamples)
      yield* lossGauge.set(avgLoss)
      yield* epochCounter.inc()
      yield* info(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`, {
        epoch,
        loss: avgLoss,
        examples: totalExamples,
        durationMs: epochResult.durationMs
      })
    })

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      yield* runEpoch(epoch)
    }
  }).pipe(Effect.withSpan("Train.trainWithStreamFactory"))

export const train = (
  examples: ReadonlyArray<string>
): Effect.Effect<void, TrainingErrorType, TrainEnv> =>
  Effect.gen(function* () {
    const llm = yield* LLMService
    const config = yield* TrainingConfig

    const endTokenId = llm.vocab.encode("</s>")
    if (endTokenId._tag === "None") {
      return yield* TrainingError.config("End token </s> not found in vocabulary")
    }

    const clipNorm = config.clipNorm ?? 5.0
    const epochCounter = yield* counter("epochs_completed")
    const lossGauge = yield* gauge("epoch_loss")
    const examplesCounter = yield* counter("examples_processed")

    const preprocessed = examples.flatMap((text) => {
      const result = preprocessExample(text, llm)
      return Option.isSome(result) ? [result.value] : []
    })

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      const epochResult = yield* timed(`epoch_${epoch}`, Effect.gen(function* () {
        let totalLoss = 0
        for (const example of preprocessed) {
          totalLoss += yield* runTrainingExample(llm, example, config.learningRate, clipNorm, endTokenId.value)
        }
        const totalExamples = preprocessed.length
        return { totalLoss, totalExamples }
      }).pipe(Effect.withSpan("Train.epochLoop")))

      const { totalLoss, totalExamples } = epochResult.value
      const avgLoss = totalExamples > 0 ? totalLoss / totalExamples : 0

      yield* examplesCounter.inc(totalExamples)
      yield* lossGauge.set(avgLoss)
      yield* epochCounter.inc()
      yield* info(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`, {
        epoch,
        loss: avgLoss,
        examples: totalExamples,
        durationMs: epochResult.durationMs
      })
    }
  }).pipe(Effect.withSpan("Train.trainArray"))

export const trainStream = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> => trainWithStreamFactory(makeStream)
