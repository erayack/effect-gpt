import { afterEach, describe, expect, test } from "bun:test"
import * as Cause from "effect/Cause"
import * as Effect from "effect/Effect"
import { AppConfig, AppConfigLive } from "../../src/config"

const configEnvVars = [
  "EFFECT_GPT_MAX_SEQ_LEN",
  "EFFECT_GPT_EMBEDDING_DIM",
  "EFFECT_GPT_HIDDEN_DIM",
  "EFFECT_GPT_TRANSFORMER_BLOCKS",
  "EFFECT_GPT_PRETRAINING_PATH",
  "EFFECT_GPT_CHAT_PATH",
  "EFFECT_GPT_DATASET_FORMAT",
  "EFFECT_GPT_PRETRAIN_EPOCHS",
  "EFFECT_GPT_PRETRAIN_LR",
  "EFFECT_GPT_FINETUNE_EPOCHS",
  "EFFECT_GPT_FINETUNE_LR"
] as const

const originalEnv = new Map<string, string | undefined>(
  configEnvVars.map((name) => [name, process.env[name]])
)

const resetConfigEnv = () => {
  for (const name of configEnvVars) {
    const originalValue = originalEnv.get(name)
    if (originalValue === undefined) {
      delete process.env[name]
    } else {
      process.env[name] = originalValue
    }
  }
}

const withConfigEnv = (values: Record<string, string>, run: () => void) => {
  resetConfigEnv()
  for (const [name, value] of Object.entries(values)) {
    process.env[name] = value
  }
  run()
}

const loadConfigExit = () =>
  Effect.runSyncExit(
    Effect.gen(function* () {
      return yield* AppConfig
    }).pipe(Effect.provide(AppConfigLive))
  )

afterEach(() => {
  resetConfigEnv()
})

describe("AppConfig", () => {
  test("loads defaults when env vars are unset", () => {
    withConfigEnv({}, () => {
      const exit = loadConfigExit()

      expect(exit._tag).toBe("Success")
      if (exit._tag === "Success") {
        expect(exit.value.model.maxSeqLen).toBe(80)
        expect(exit.value.model.embeddingDim).toBe(128)
        expect(exit.value.model.hiddenDim).toBe(256)
        expect(exit.value.model.transformerBlocks).toBe(3)
        expect(exit.value.dataset.pretrainingPath).toBe("data/pretraining_data.json")
        expect(exit.value.dataset.chatPath).toBe("data/chat_training_data.json")
        expect(exit.value.dataset.format).toBe("json")
        expect(exit.value.training.pretraining.epochs).toBe(100)
        expect(exit.value.training.pretraining.learningRate).toBe(0.0005)
        expect(exit.value.training.finetuning.epochs).toBe(100)
        expect(exit.value.training.finetuning.learningRate).toBe(0.0001)
      }
    })
  })

  test("loads and parses env var overrides", () => {
    withConfigEnv(
      {
        EFFECT_GPT_MAX_SEQ_LEN: "64",
        EFFECT_GPT_EMBEDDING_DIM: "96",
        EFFECT_GPT_HIDDEN_DIM: "192",
        EFFECT_GPT_TRANSFORMER_BLOCKS: "2",
        EFFECT_GPT_PRETRAINING_PATH: "data/custom_pretrain.json",
        EFFECT_GPT_CHAT_PATH: "data/custom_chat.json",
        EFFECT_GPT_DATASET_FORMAT: "csv",
        EFFECT_GPT_PRETRAIN_EPOCHS: "25",
        EFFECT_GPT_PRETRAIN_LR: "0.001",
        EFFECT_GPT_FINETUNE_EPOCHS: "15",
        EFFECT_GPT_FINETUNE_LR: "0.0002"
      },
      () => {
        const exit = loadConfigExit()

        expect(exit._tag).toBe("Success")
        if (exit._tag === "Success") {
          expect(exit.value.model.maxSeqLen).toBe(64)
          expect(exit.value.model.embeddingDim).toBe(96)
          expect(exit.value.model.hiddenDim).toBe(192)
          expect(exit.value.model.transformerBlocks).toBe(2)
          expect(exit.value.dataset.pretrainingPath).toBe("data/custom_pretrain.json")
          expect(exit.value.dataset.chatPath).toBe("data/custom_chat.json")
          expect(exit.value.dataset.format).toBe("csv")
          expect(exit.value.training.pretraining.epochs).toBe(25)
          expect(exit.value.training.pretraining.learningRate).toBe(0.001)
          expect(exit.value.training.finetuning.epochs).toBe(15)
          expect(exit.value.training.finetuning.learningRate).toBe(0.0002)
        }
      }
    )
  })

  test("fails fast when numeric validations are violated", () => {
    withConfigEnv({ EFFECT_GPT_TRANSFORMER_BLOCKS: "0" }, () => {
      const exit = loadConfigExit()

      expect(exit._tag).toBe("Failure")
      if (exit._tag === "Failure") {
        const message = Cause.pretty(exit.cause)
        expect(message).toContain("EFFECT_GPT_TRANSFORMER_BLOCKS")
        expect(message).toContain("positive integer")
      }
    })
  })
})
