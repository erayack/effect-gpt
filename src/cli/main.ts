import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as HashSet from "effect/HashSet"
import { Terminal } from "@effect/platform"
import { BunFileSystem, BunRuntime, BunTerminal } from "@effect/platform-bun"
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
  makePreprocessSettingsLayer
} from "../training/train"
import { MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import { TerminalLoggerLive, info, error as logError } from "../services/Logger"
import { InMemoryMetricsLive, snapshot } from "../services/Metrics"

const PRETRAIN_EPOCHS = 100
const PRETRAIN_LR = 0.0005
const FINETUNE_EPOCHS = 100
const FINETUNE_LR = 0.0001

const readLine = (prompt: string) =>
  Effect.gen(function* () {
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
    })
  )

const main = Effect.scoped(
  Effect.gen(function* () {
    const terminal = yield* Terminal.Terminal

    const dataset = Dataset.load({
      pretrainingPath: "data/pretraining_data.json",
      chatPath: "data/chat_training_data.json",
      format: "json"
    })

    const vocabSet1 = yield* Vocab.processStreamForVocab(dataset.pretrainingStream())
    const vocabSet2 = yield* Vocab.processStreamForVocab(dataset.chatStream())

    const combinedSet = HashSet.union(vocabSet1, vocabSet2)
    const vocabWords = Array.from(HashSet.values(combinedSet)).sort()
    const vocab = Vocab.make(vocabWords)

    const vocabSize = vocab.words.length
    const network = [
      new Embeddings(vocabSize, EMBEDDING_DIM, MAX_SEQ_LEN),
      new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM),
      new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM),
      new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM),
      new OutputProjection(EMBEDDING_DIM, vocabSize)
    ]
    const llm = new LLM(vocab, network)

    yield* terminal.display("\n=== MODEL INFORMATION ===\n")
    yield* terminal.display(`Network architecture: ${llm.networkDescription()}\n`)
    yield* terminal.display(
      `Model configuration -> max_seq_len: ${MAX_SEQ_LEN}, embedding_dim: ${EMBEDDING_DIM}, hidden_dim: ${HIDDEN_DIM}\n`
    )
    yield* terminal.display(`Total parameters: ${llm.totalParameters()}\n`)

    const testInput = "User: How do mountains form?"

    yield* terminal.display("\n=== BEFORE TRAINING ===\n")
    yield* terminal.display(`Input: ${testInput}\n`)
    const beforeOutput = yield* llm.predict(testInput)
    yield* terminal.display(`Output: ${beforeOutput}\n`)

    const llmLayer = makeLLMLayer(llm)
    const preprocessLayer = makePreprocessSettingsLayer({ concurrency: "unbounded", batchSize: 1 })

    yield* info("\n=== PRE-TRAINING MODEL ===")
    yield* info(`Pre-training for ${PRETRAIN_EPOCHS} epochs with learning rate ${PRETRAIN_LR}`)
    yield* trainStream(dataset.pretrainingStream).pipe(
      Effect.provide(llmLayer),
      Effect.provide(makeTrainingConfigLayer({ epochs: PRETRAIN_EPOCHS, learningRate: PRETRAIN_LR })),
      Effect.provide(preprocessLayer)
    )

    yield* info("\n=== INSTRUCTION TUNING ===")
    yield* info(`Instruction tuning for ${FINETUNE_EPOCHS} epochs with learning rate ${FINETUNE_LR}`)
    yield* trainStream(dataset.chatStream).pipe(
      Effect.provide(llmLayer),
      Effect.provide(makeTrainingConfigLayer({ epochs: FINETUNE_EPOCHS, learningRate: FINETUNE_LR })),
      Effect.provide(preprocessLayer)
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
  })
)

const LoggerLayer = TerminalLoggerLive("info").pipe(Layer.provide(BunTerminal.layer))

const AppLayer = Layer.mergeAll(BunFileSystem.layer, BunTerminal.layer, LoggerLayer, InMemoryMetricsLive)

const program = Effect.scoped(
  main.pipe(
    Effect.provide(AppLayer),
    Effect.catchAll((err) => logError(`Fatal: ${String(err)}`).pipe(Effect.provide(AppLayer)))
  )
)

BunRuntime.runMain(program)
