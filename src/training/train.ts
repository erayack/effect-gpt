import * as Effect from "effect/Effect"
import * as Stream from "effect/Stream"
import * as Chunk from "effect/Chunk"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import type { ShapeError } from "../tensor/ops"
import * as Ops from "../tensor/ops"
import * as T from "../tensor/Tensor2D"
import { tokenize } from "../tokenize/tokenize"
import type { LLM } from "../model/LLM"
import { softmaxRows, crossEntropyLoss, dLogits } from "./loss"
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
  readonly preprocessConcurrency?: number | "unbounded"
  readonly preprocessBatchSize?: number
}

interface TrainingConfigId {
  readonly TrainingConfig: unique symbol
}
interface LLMServiceId {
  readonly LLMService: unique symbol
}

export interface PreprocessSettings {
  readonly concurrency: number | "unbounded"
  readonly batchSize: number
}

interface PreprocessSettingsId {
  readonly PreprocessSettings: unique symbol
}

export const TrainingConfig = Context.GenericTag<TrainingConfigId, TrainingConfig>("TrainingConfig")
export const LLMService = Context.GenericTag<LLMServiceId, LLM>("LLMService")
export const PreprocessSettings = Context.GenericTag<PreprocessSettingsId, PreprocessSettings>(
  "PreprocessSettings"
)

export const makeLLMLayer = (llm: LLM) => Layer.succeed(LLMService, llm)
export const makeTrainingConfigLayer = (config: TrainingConfig) =>
  Layer.succeed(TrainingConfig, config)
export const makePreprocessSettingsLayer = (settings: PreprocessSettings) =>
  Layer.succeed(PreprocessSettings, settings)

type TrainEnv =
  | TrainingConfigId
  | LLMServiceId
  | LoggerServiceId
  | MetricsServiceId
  | PreprocessSettingsId

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

const trainWithStreamFactory = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> =>
  Effect.gen(function* () {
    const llm = yield* LLMService
    const config = yield* TrainingConfig
    const preprocessSettings = yield* Effect.gen(function* () {
      const env = (yield* Effect.context<R | TrainEnv>()) as Context.Context<R | TrainEnv>
      const maybeSettings = Context.getOption(env, PreprocessSettings)
      if (maybeSettings._tag === "Some") {
        return maybeSettings.value
      }
      return {
        concurrency: config.preprocessConcurrency ?? "unbounded",
        batchSize: config.preprocessBatchSize ?? 1
      } satisfies PreprocessSettings
    })

    const endTokenId = llm.vocab.encode("</s>")
    if (endTokenId._tag === "None") {
      return yield* Effect.fail(TrainingError.config("End token </s> not found in vocabulary"))
    }

    const clipNorm = config.clipNorm ?? 5.0
    const concurrency = preprocessSettings.concurrency
    const batchSize = Math.max(1, preprocessSettings.batchSize)

    const epochCounter = yield* counter("epochs_completed")
    const lossGauge = yield* gauge("epoch_loss")
    const examplesCounter = yield* counter("examples_processed")

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      const epochResult = yield* timed(`epoch_${epoch}`, Effect.gen(function* () {
        let totalLoss = 0
        let totalExamples = 0

        const preprocess = (text: string) =>
          Effect.sync(() => {
            totalExamples++
            const tokens = [...tokenize(text, llm.vocab)]
            if (tokens.length < 2) {
              return Option.none<{ inputIds: number[]; targetIds: number[] }>()
            }

            return Option.some({
              inputIds: tokens.slice(0, tokens.length - 1),
              targetIds: tokens.slice(1)
            })
        })

        const preprocessed = makeStream()
          .pipe(
            Stream.mapError(TrainingError.fromUnknown),
            Stream.mapChunks(Chunk.chunksOf(batchSize)),
            Stream.flattenChunks,
            Stream.mapEffect(preprocess, { concurrency }),
            Stream.filterMap((value) => value)
          )

        yield* Effect.scoped(
          Stream.runForEachScoped(preprocessed, ({ inputIds, targetIds }) =>
            Effect.gen(function* () {
              let input = T.fromArray(1, inputIds.length, inputIds)
              for (const layer of llm.network) {
                input = yield* mapShapeError(layer.forward(input))
              }

              const logits = input
              const probs = yield* wrapThrowing(() => softmaxRows(logits), mapShapeUnknown)

              const loss = yield* wrapThrowing(() => crossEntropyLoss(probs, targetIds), mapShapeUnknown)
              totalLoss += loss
              let grads = yield* wrapThrowing(() => dLogits(probs, targetIds), mapShapeUnknown)
              clipGlobalL2(grads, clipNorm)

              for (let i = llm.network.length - 1; i >= 0; i--) {
                grads = yield* mapShapeError(llm.network[i]!.backward(grads, config.learningRate))
              }

              const tokens = Ops.argmaxRows(probs)
              const nextToken = tokens[tokens.length - 1]
              if (nextToken === endTokenId.value) {
                return
              }
            })
          )
        )

        yield* examplesCounter.inc(totalExamples)
        return { totalLoss, totalExamples }
      }))

      const { totalLoss, totalExamples } = epochResult.value
      const avgLoss = totalExamples > 0 ? totalLoss / totalExamples : 0

      yield* lossGauge.set(avgLoss)
      yield* epochCounter.inc()
      yield* info(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`, {
        epoch,
        loss: avgLoss,
        examples: totalExamples,
        durationMs: epochResult.durationMs
      })
    }
  })

export const train = (
  examples: ReadonlyArray<string>
): Effect.Effect<void, TrainingErrorType, TrainEnv> =>
  trainWithStreamFactory(() => Stream.fromIterable(examples))

export const trainStream = <E, R>(
  makeStream: () => Stream.Stream<string, E, R>
): Effect.Effect<void, TrainingErrorType, R | TrainEnv> => trainWithStreamFactory(makeStream)
