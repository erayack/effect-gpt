import * as Effect from "effect/Effect"
import * as HashSet from "effect/HashSet"
import * as Layer from "effect/Layer"
import { EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN } from "../src/config"
import { Embeddings } from "../src/model/Embeddings"
import { LLM } from "../src/model/LLM"
import { OutputProjection } from "../src/model/OutputProjection"
import { TransformerBlock } from "../src/model/TransformerBlock"
import { InMemoryMetricsLive, snapshot } from "../src/services/Metrics"
import { SilentLoggerLive } from "../src/services/Logger"
import { seeded } from "../src/tensor/random"
import { tokenize } from "../src/tokenize/tokenize"
import {
  makeLLMLayer,
  makePreprocessSettingsLayer,
  makeTrainingConfigLayer,
  train
} from "../src/training/train"
import { Vocab } from "../src/vocab/Vocab"

type OutputFormat = "text" | "json"

interface BenchmarkOptions {
  readonly scenarioNames: ReadonlyArray<string>
  readonly iterations: number
  readonly warmup: number
  readonly format: OutputFormat
  readonly outputPath?: string
}

interface BenchmarkRun {
  readonly durationMs: number
  readonly metrics: Readonly<Record<string, number>>
}

interface BenchmarkMetricSummary {
  readonly min: number
  readonly max: number
  readonly mean: number
  readonly median: number
  readonly p95: number
}

interface ScenarioSummary {
  readonly name: string
  readonly description: string
  readonly metadata: Readonly<Record<string, string | number>>
  readonly runs: ReadonlyArray<BenchmarkRun>
  readonly metrics: Readonly<Record<string, BenchmarkMetricSummary>>
}

interface BenchmarkSuiteResult {
  readonly generatedAt: string
  readonly iterations: number
  readonly warmup: number
  readonly scenarios: ReadonlyArray<ScenarioSummary>
}

interface BenchmarkScenario {
  readonly name: string
  readonly description: string
  readonly metadata: Readonly<Record<string, string | number>>
  readonly run: () => Promise<BenchmarkRun>
}

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

const repeatedCorpus = (repeats: number): ReadonlyArray<string> =>
  Array.from({ length: repeats }, (_, repeat) =>
    corpusBase.map((line, index) => `${line} sample_${repeat}_${index}`)
  ).flat()

const trainingCorpus = repeatedCorpus(16)
const longPrompt = trainingCorpus.slice(0, 6).map((line) => line.replace(" </s>", "")).join(" ")
const benchmarkTexts = [...trainingCorpus, "user: explain gradients simply", longPrompt]

const buildVocab = (): Vocab => {
  const vocabWords = Array.from(HashSet.values(Vocab.processTextForVocab(benchmarkTexts))).sort()
  return Vocab.make(vocabWords)
}

const benchmarkVocab = buildVocab()

const createBenchmarkLLM = (transformerBlocks = 2): LLM => {
  const rng = seeded(42)
  return new LLM(benchmarkVocab, [
    new Embeddings(benchmarkVocab.words.length, EMBEDDING_DIM, MAX_SEQ_LEN, rng),
    ...Array.from({ length: transformerBlocks }, () => new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM, rng)),
    new OutputProjection(EMBEDDING_DIM, benchmarkVocab.words.length, rng)
  ])
}

const median = (values: ReadonlyArray<number>): number => {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!
}

const percentile = (values: ReadonlyArray<number>, p: number): number => {
  const sorted = [...values].sort((a, b) => a - b)
  const clamped = Math.min(1, Math.max(0, p))
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * clamped) - 1)
  return sorted[Math.max(0, index)]!
}

const summarizeMetric = (values: ReadonlyArray<number>): BenchmarkMetricSummary => {
  const sum = values.reduce((total, value) => total + value, 0)
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: sum / values.length,
    median: median(values),
    p95: percentile(values, 0.95)
  }
}

const formatNumber = (value: number): string =>
  Number.isInteger(value) ? value.toString() : value.toFixed(value >= 100 ? 2 : 3)

const collectSummaries = (runs: ReadonlyArray<BenchmarkRun>): Readonly<Record<string, BenchmarkMetricSummary>> => {
  const metricNames = new Set<string>()
  for (const run of runs) {
    for (const metricName of Object.keys(run.metrics)) {
      metricNames.add(metricName)
    }
  }

  const summaries: Record<string, BenchmarkMetricSummary> = {}
  for (const metricName of metricNames) {
    summaries[metricName] = summarizeMetric(runs.map((run) => run.metrics[metricName] ?? Number.NaN))
  }

  return summaries
}

const buildTrainingScenario = (): BenchmarkScenario => {
  const epochs = 2
  const batchSize = 4
  const trainableTokensPerEpoch = trainingCorpus.reduce((total, text) => {
    const tokenCount = tokenize(text, benchmarkVocab).length
    return total + Math.max(0, tokenCount - 1)
  }, 0)
  const totalTrainableTokens = trainableTokensPerEpoch * epochs

  return {
    name: "train-small",
    description: "End-to-end minibatch training throughput on the benchmark corpus.",
    metadata: {
      corpus_examples: trainingCorpus.length,
      epochs,
      batch_size: batchSize,
      transformer_blocks: 2,
      trainable_tokens_per_run: totalTrainableTokens
    },
    run: async () => {
      const llm = createBenchmarkLLM(2)
      const trainingLayer = Layer.mergeAll(
        SilentLoggerLive,
        InMemoryMetricsLive,
        makeLLMLayer(llm),
        makeTrainingConfigLayer({
          epochs,
          learningRate: 0.0005,
          clipNorm: 5,
          trainConcurrency: 1
        }),
        makePreprocessSettingsLayer({
          concurrency: 1,
          batchSize,
          cacheScope: "perRun"
        })
      )

      const started = performance.now()
      const metricsSnapshot = await Effect.runPromise(
        Effect.gen(function* () {
          yield* train(trainingCorpus)
          return yield* snapshot()
        }).pipe(Effect.provide(trainingLayer))
      )
      const durationMs = performance.now() - started
      const epochDurations = metricsSnapshot.timings
        .filter((timing) => timing.label.startsWith("epoch_"))
        .map((timing) => timing.durationMs)
      const meanEpochMs = epochDurations.length === 0
        ? Number.NaN
        : epochDurations.reduce((sum, value) => sum + value, 0) / epochDurations.length
      const finalLoss = metricsSnapshot.gauges.find((gauge) => gauge.name === "epoch_loss")?.value ?? Number.NaN

      return {
        durationMs,
        metrics: {
          duration_ms: durationMs,
          epoch_ms: meanEpochMs,
          final_loss: finalLoss,
          trainable_tokens: totalTrainableTokens,
          trainable_tokens_per_sec: (totalTrainableTokens / durationMs) * 1000
        }
      }
    }
  }
}

const buildInferenceScenario = (
  name: string,
  description: string,
  prompt: string
): BenchmarkScenario => {
  const llm = createBenchmarkLLM(2)
  const promptTokens = tokenize(prompt, llm.vocab).length

  return {
    name,
    description,
    metadata: {
      prompt_tokens: promptTokens,
      transformer_blocks: 2,
      max_seq_len: MAX_SEQ_LEN
    },
    run: async () => {
      const started = performance.now()
      const outputTokens = await Effect.runPromise(llm.forward(prompt))
      const durationMs = performance.now() - started
      const generatedTokens = outputTokens.length
      const totalTokens = promptTokens + generatedTokens

      return {
        durationMs,
        metrics: {
          duration_ms: durationMs,
          prompt_tokens: promptTokens,
          generated_tokens: generatedTokens,
          total_tokens: totalTokens,
          generated_tokens_per_sec: generatedTokens === 0 ? 0 : (generatedTokens / durationMs) * 1000,
          total_tokens_per_sec: totalTokens === 0 ? 0 : (totalTokens / durationMs) * 1000,
          ms_per_generated_token: generatedTokens === 0 ? 0 : durationMs / generatedTokens
        }
      }
    }
  }
}

export const defaultBenchmarkScenarios = (): ReadonlyArray<BenchmarkScenario> => [
  buildTrainingScenario(),
  buildInferenceScenario(
    "generate-short",
    "Greedy generation from a short prompt to track decode throughput.",
    "user: explain gradients simply"
  ),
  buildInferenceScenario(
    "generate-long",
    "Greedy generation from a longer prompt to expose prefill plus decode cost.",
    longPrompt
  )
]

const parseArgs = (argv: ReadonlyArray<string>): BenchmarkOptions => {
  const args = [...argv]
  let iterations = 5
  let warmup = 1
  let format: OutputFormat = "text"
  let outputPath: string | undefined
  const scenarioNames: Array<string> = []

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    const next = args[i + 1]

    switch (arg) {
      case "--scenario":
      case "-s":
        if (next === undefined) {
          throw new Error("Missing value for --scenario")
        }
        scenarioNames.push(next)
        i += 1
        break
      case "--iterations":
      case "-n":
        if (next === undefined) {
          throw new Error("Missing value for --iterations")
        }
        iterations = Number.parseInt(next, 10)
        i += 1
        break
      case "--warmup":
      case "-w":
        if (next === undefined) {
          throw new Error("Missing value for --warmup")
        }
        warmup = Number.parseInt(next, 10)
        i += 1
        break
      case "--format":
      case "-f":
        if (next !== "text" && next !== "json") {
          throw new Error("Format must be 'text' or 'json'")
        }
        format = next
        i += 1
        break
      case "--output":
      case "-o":
        if (next === undefined) {
          throw new Error("Missing value for --output")
        }
        outputPath = next
        i += 1
        break
      case "--help":
      case "-h":
        printUsage()
        process.exit(0)
      default:
        throw new Error(`Unknown argument: ${arg}`)
    }
  }

  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("--iterations must be a positive integer")
  }
  if (!Number.isInteger(warmup) || warmup < 0) {
    throw new Error("--warmup must be a non-negative integer")
  }

  return {
    scenarioNames,
    iterations,
    warmup,
    format,
    ...(outputPath === undefined ? {} : { outputPath })
  }
}

const printUsage = (): void => {
  console.log("Usage: bun benchmarks/harness.ts [--scenario NAME] [--iterations N] [--warmup N] [--format text|json] [--output PATH]")
  console.log("Available scenarios: train-small, generate-short, generate-long")
}

const resolveScenarios = (requestedNames: ReadonlyArray<string>): ReadonlyArray<BenchmarkScenario> => {
  const byName = new Map(defaultBenchmarkScenarios().map((scenario) => [scenario.name, scenario]))
  if (requestedNames.length === 0) {
    return [...byName.values()]
  }

  return requestedNames.map((name) => {
    const scenario = byName.get(name)
    if (scenario === undefined) {
      throw new Error(`Unknown scenario '${name}'. Available scenarios: ${[...byName.keys()].join(", ")}`)
    }
    return scenario
  })
}

export const runBenchmarkSuite = async (options: BenchmarkOptions): Promise<BenchmarkSuiteResult> => {
  const scenarios = resolveScenarios(options.scenarioNames)
  const summaries: Array<ScenarioSummary> = []

  for (const scenario of scenarios) {
    for (let i = 0; i < options.warmup; i++) {
      await scenario.run()
    }

    const runs: Array<BenchmarkRun> = []
    for (let i = 0; i < options.iterations; i++) {
      runs.push(await scenario.run())
    }

    summaries.push({
      name: scenario.name,
      description: scenario.description,
      metadata: scenario.metadata,
      runs,
      metrics: collectSummaries(runs)
    })
  }

  return {
    generatedAt: new Date().toISOString(),
    iterations: options.iterations,
    warmup: options.warmup,
    scenarios: summaries
  }
}

const renderTextReport = (result: BenchmarkSuiteResult): string => {
  const lines = [
    `Benchmark suite generated at ${result.generatedAt}`,
    `iterations=${result.iterations} warmup=${result.warmup}`
  ]

  for (const scenario of result.scenarios) {
    lines.push("")
    lines.push(`[${scenario.name}] ${scenario.description}`)
    lines.push(`metadata ${Object.entries(scenario.metadata).map(([key, value]) => `${key}=${value}`).join(" ")}`)
    for (const [metricName, summary] of Object.entries(scenario.metrics)) {
      lines.push(
        `${metricName} median=${formatNumber(summary.median)} mean=${formatNumber(summary.mean)} min=${formatNumber(summary.min)} max=${formatNumber(summary.max)} p95=${formatNumber(summary.p95)}`
      )
    }
  }

  return lines.join("\n")
}

export const runCli = async (argv: ReadonlyArray<string> = process.argv.slice(2)): Promise<void> => {
  const options = parseArgs(argv)
  const result = await runBenchmarkSuite(options)
  const serialized = JSON.stringify(result, null, 2)

  if (options.outputPath !== undefined) {
    await Bun.write(options.outputPath, serialized)
  }

  if (options.format === "json") {
    console.log(serialized)
    return
  }

  console.log(renderTextReport(result))
}

if (import.meta.main) {
  await runCli()
}
