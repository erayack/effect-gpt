import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as HashSet from "effect/HashSet"
import { makeLLMLayer, makeTrainingConfigLayer, makePreprocessSettingsLayer, train } from "../src/training/train"
import { LLM } from "../src/model/LLM"
import { Embeddings } from "../src/model/Embeddings"
import { TransformerBlock } from "../src/model/TransformerBlock"
import { OutputProjection } from "../src/model/OutputProjection"
import { Vocab } from "../src/vocab/Vocab"
import { seeded } from "../src/tensor/random"
import { SilentLoggerLive } from "../src/services/Logger"
import { InMemoryMetricsLive, snapshot } from "../src/services/Metrics"
import { EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN } from "../src/config"

const corpusBase = [
  "user: explain gradients simply </s>",
  "assistant: gradients show how each weight should change </s>",
  "user: what is attention in a transformer </s>",
  "assistant: attention lets each token weigh other tokens </s>",
  "user: why use layer norm </s>",
  "assistant: layer norm keeps activations stable during training </s>",
  "user: what does backpropagation do </s>",
  "assistant: backpropagation sends loss information backward through the network </s>"
] as const

const corpus = Array.from({ length: 16 }, (_, repeat) =>
  corpusBase.map((line, index) => `${line} sample_${repeat}_${index}`)
).flat()

const buildVocab = () => {
  const vocabWords = Array.from(HashSet.values(Vocab.processTextForVocab(corpus))).sort()
  return Vocab.make(vocabWords)
}

const makeLlm = () => {
  const vocab = buildVocab()
  const rng = seeded(42)
  return new LLM(vocab, [
    new Embeddings(vocab.words.length, EMBEDDING_DIM, MAX_SEQ_LEN, rng),
    new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM, rng),
    new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM, rng),
    new OutputProjection(EMBEDDING_DIM, vocab.words.length, rng)
  ])
}

interface RunResult {
  readonly totalMs: number
  readonly meanEpochMs: number
  readonly finalLoss: number
}

const median = (values: ReadonlyArray<number>): number => {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!
}

const benchmarkOnce = async (): Promise<RunResult> => {
  const llm = makeLlm()
  const trainingLayer = Layer.mergeAll(
    SilentLoggerLive,
    InMemoryMetricsLive,
    makeLLMLayer(llm),
    makeTrainingConfigLayer({
      epochs: 2,
      learningRate: 0.0005,
      clipNorm: 5,
      trainConcurrency: 1
    }),
    makePreprocessSettingsLayer({
      concurrency: 1,
      batchSize: 4,
      cacheScope: "perRun"
    })
  )

  const started = performance.now()
  const { metrics } = await Effect.runPromise(
    Effect.gen(function* () {
      yield* train(corpus)
      return { metrics: yield* snapshot() }
    }).pipe(Effect.provide(trainingLayer))
  )
  const totalMs = performance.now() - started
  const epochDurations = metrics.timings
    .filter((timing) => timing.label.startsWith("epoch_"))
    .map((timing) => timing.durationMs)
  const finalLoss = metrics.gauges.find((gauge) => gauge.name === "epoch_loss")?.value ?? Number.NaN
  const meanEpochMs = epochDurations.length === 0
    ? Number.NaN
    : epochDurations.reduce((sum, value) => sum + value, 0) / epochDurations.length

  return { totalMs, meanEpochMs, finalLoss }
}

const main = async () => {
  await benchmarkOnce()

  const runs = [] as Array<RunResult>
  for (let i = 0; i < 5; i++) {
    runs.push(await benchmarkOnce())
  }

  const totalMs = median(runs.map((run) => run.totalMs))
  const meanEpochMs = median(runs.map((run) => run.meanEpochMs))
  const finalLoss = median(runs.map((run) => run.finalLoss))

  console.log(`METRIC total_ms=${totalMs.toFixed(3)}`)
  console.log(`METRIC epoch_ms=${meanEpochMs.toFixed(3)}`)
  console.log(`METRIC final_loss=${finalLoss.toFixed(6)}`)
  console.log(`METRIC corpus_examples=${corpus.length}`)
}

await main()
