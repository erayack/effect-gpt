import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as HashSet from "effect/HashSet"
import * as Option from "effect/Option"
import { Terminal } from "@effect/platform"
import { Command, Options } from "@effect/cli"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import pkg from "../../package.json" with { type: "json" }
import { Dataset } from "../data/Dataset"
import { Vocab } from "../vocab/Vocab"
import { LLM } from "../model/LLM"
import { Embeddings } from "../model/Embeddings"
import { TransformerBlock } from "../model/TransformerBlock"
import { OutputProjection } from "../model/OutputProjection"
import {
  trainStream,
  makeLLMLayer,
  makeTrainingConfigLayer,
  CachedPreprocessSettingsLive
} from "../training/train"
import { AppConfig, AppConfigLive } from "../config"
import { PrettyLoggerLive, info } from "../services/Logger"
import { InMemoryMetricsLive, snapshot } from "../services/Metrics"
import { SeedLayer, useSeedRng } from "../services/SeedLayer"
import type { Rng } from "../tensor/random"
import { withCliErrorLogging } from "./program"

const readLine = Effect.fn("Cli.readLine")(function* (prompt: string) {
    const terminal = yield* Terminal.Terminal
    yield* terminal.display(prompt)
    return yield* terminal.readLine
  })

const repl = (llm: LLM) =>
  Effect.scoped(
    Effect.gen(function* () {
      const terminal = yield* Terminal.Terminal
      yield* terminal.display("\n--- Interactive Mode ---\n")
      yield* terminal.display("Type a prompt and press Enter to generate text.\n")
      yield* terminal.display("Type 'exit' to quit.\n")

      while (true) {
        const input = yield* readLine("\nEnter prompt: ")
        const trimmed = input.trim()

        if (trimmed.toLowerCase() === "exit") {
          yield* terminal.display("Exiting interactive mode.\n")
          break
        }

        const formattedInput = `User: ${trimmed}`
        const prediction = yield* llm.predict(formattedInput)
        yield* terminal.display(`Model output: ${prediction}\n`)
      }
    }).pipe(Effect.withSpan("Cli.repl"))
  )

const main = Effect.scoped(
  Effect.gen(function* () {
    const terminal = yield* Terminal.Terminal
    const rng: Rng = yield* useSeedRng()
    const appConfig = yield* AppConfig

    const dataset = Dataset.load({
      pretrainingPath: appConfig.dataset.pretrainingPath,
      chatPath: appConfig.dataset.chatPath,
      format: appConfig.dataset.format
    })

    const vocabSet1 = yield* Vocab.processStreamForVocab(dataset.pretrainingStream())
    const vocabSet2 = yield* Vocab.processStreamForVocab(dataset.chatStream())

    const combinedSet = HashSet.union(vocabSet1, vocabSet2)
    const vocabWords = Array.from(HashSet.values(combinedSet)).sort()
    const vocab = Vocab.make(vocabWords)

    const vocabSize = vocab.words.length
    const network = [
      new Embeddings(vocabSize, appConfig.model.embeddingDim, appConfig.model.maxSeqLen, rng),
      ...Array.from(
        { length: appConfig.model.transformerBlocks },
        () => new TransformerBlock(appConfig.model.embeddingDim, appConfig.model.hiddenDim, rng)
      ),
      new OutputProjection(appConfig.model.embeddingDim, vocabSize, rng)
    ]
    const llm = new LLM(vocab, network)

    yield* terminal.display("\n=== MODEL INFORMATION ===\n")
    yield* terminal.display(`Network architecture: ${llm.networkDescription()}\n`)
    yield* terminal.display(
      `Model configuration -> max_seq_len: ${appConfig.model.maxSeqLen}, embedding_dim: ${appConfig.model.embeddingDim}, hidden_dim: ${appConfig.model.hiddenDim}\n`
    )
    yield* terminal.display(`Total parameters: ${llm.totalParameters()}\n`)

    const testInput = "User: How do mountains form?"

    yield* terminal.display("\n=== BEFORE TRAINING ===\n")
    yield* terminal.display(`Input: ${testInput}\n`)
    const beforeOutput = yield* llm.predict(testInput)
    yield* terminal.display(`Output: ${beforeOutput}\n`)

    const llmLayer = makeLLMLayer(llm)
    const baseTrainingLayer = Layer.mergeAll(llmLayer, CachedPreprocessSettingsLive)
    const runTrainingPhase = Effect.fn("Cli.runTrainingPhase")(function* (
      name: string,
      makeDatasetStream: typeof dataset.pretrainingStream,
      trainingConfig: {
        readonly epochs: number
        readonly learningRate: number
      }
    ) {
      yield* info(`\n=== ${name} ===`)
      yield* info(
        `${name} for ${trainingConfig.epochs} epochs with learning rate ${trainingConfig.learningRate}`
      )
      const trainingLayer = Layer.mergeAll(
        baseTrainingLayer,
        makeTrainingConfigLayer({
          epochs: trainingConfig.epochs,
          learningRate: trainingConfig.learningRate
        })
      )
      yield* trainStream(makeDatasetStream).pipe(Effect.provide(trainingLayer))
    })

    yield* runTrainingPhase(
      "PRE-TRAINING MODEL",
      dataset.pretrainingStream,
      appConfig.training.pretraining
    )
    yield* runTrainingPhase(
      "INSTRUCTION TUNING",
      dataset.chatStream,
      appConfig.training.finetuning
    )

    const metrics = yield* snapshot()
    yield* info("Training complete", {
      epochsCompleted: metrics.counters.find((c) => c.name === "epochs_completed")?.value,
      totalExamples: metrics.counters.find((c) => c.name === "examples_processed")?.value,
      finalLoss: metrics.gauges.find((g) => g.name === "epoch_loss")?.value
    })

    yield* terminal.display("\n=== AFTER TRAINING ===\n")
    yield* terminal.display(`Input: ${testInput}\n`)
    const afterOutput = yield* llm.predict(testInput)
    yield* terminal.display(`Output: ${afterOutput}\n`)
    yield* terminal.display("======================\n")

    yield* repl(llm)
  }).pipe(Effect.withSpan("Cli.main"))
)

const LoggerLayer = PrettyLoggerLive("info")

const makeAppLayer = (seed?: number) =>
  Layer.mergeAll(
    LoggerLayer,
    InMemoryMetricsLive,
    SeedLayer(seed),
    AppConfigLive
  )

const runTrainingProgram = (seed?: number) =>
  withCliErrorLogging(main).pipe(Effect.provide(makeAppLayer(seed)))

const seedOption = Options.integer("seed").pipe(
  Options.optional,
  Options.withDescription("Optional deterministic seed for model initialization.")
)

const trainCommand = Command.make(
  "effect-gpt",
  { seed: seedOption },
  Effect.fn("Cli.trainCommand")(function* ({ seed }) {
    yield* runTrainingProgram(Option.getOrUndefined(seed))
  })
).pipe(
  Command.withDescription("Train and run the Effect GPT model.")
)

const cli = Command.run(trainCommand, {
  name: "effect-gpt",
  version: pkg.version
})

export const makeProgram = (argv: ReadonlyArray<string>) =>
  cli(argv).pipe(Effect.provide(BunContext.layer))

if (import.meta.main) {
  BunRuntime.runMain(makeProgram(process.argv))
}
