import * as Config from "effect/Config"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"

export const MAX_SEQ_LEN = 80
export const EMBEDDING_DIM = 128
export const HIDDEN_DIM = 256

export interface ModelConfig {
  readonly maxSeqLen: number
  readonly embeddingDim: number
  readonly hiddenDim: number
  readonly transformerBlocks: number
}

export interface DatasetConfig {
  readonly pretrainingPath: string
  readonly chatPath: string
  readonly format: "json" | "csv"
}

export interface TrainingPhaseConfig {
  readonly epochs: number
  readonly learningRate: number
}

export interface AppConfig {
  readonly model: ModelConfig
  readonly dataset: DatasetConfig
  readonly training: {
    readonly pretraining: TrainingPhaseConfig
    readonly finetuning: TrainingPhaseConfig
  }
}

class AppConfigTag extends Context.Tag("effect-gpt/config/AppConfig")<AppConfigTag, AppConfig>() {}

export const AppConfig = AppConfigTag

const positiveIntegerConfig = (name: string, defaultValue: number, description: string) =>
  Config.integer(name).pipe(
    Config.withDefault(defaultValue),
    Config.validate({
      message: `${name} must be a positive integer`,
      validation: (value) => value > 0
    }),
    Config.withDescription(description)
  )

const positiveNumberConfig = (name: string, defaultValue: number, description: string) =>
  Config.number(name).pipe(
    Config.withDefault(defaultValue),
    Config.validate({
      message: `${name} must be a positive number`,
      validation: (value) => Number.isFinite(value) && value > 0
    }),
    Config.withDescription(description)
  )

const nonEmptyStringConfig = (name: string, defaultValue: string, description: string) =>
  Config.string(name).pipe(
    Config.withDefault(defaultValue),
    Config.validate({
      message: `${name} must be a non-empty string`,
      validation: (value) => value.trim().length > 0
    }),
    Config.withDescription(description)
  )

const appConfig = Config.all({
  model: Config.all({
    maxSeqLen: positiveIntegerConfig(
      "EFFECT_GPT_MAX_SEQ_LEN",
      MAX_SEQ_LEN,
      "Maximum token sequence length."
    ),
    embeddingDim: positiveIntegerConfig(
      "EFFECT_GPT_EMBEDDING_DIM",
      EMBEDDING_DIM,
      "Transformer embedding dimension."
    ),
    hiddenDim: positiveIntegerConfig(
      "EFFECT_GPT_HIDDEN_DIM",
      HIDDEN_DIM,
      "Hidden layer dimension for feed-forward blocks."
    ),
    transformerBlocks: positiveIntegerConfig(
      "EFFECT_GPT_TRANSFORMER_BLOCKS",
      3,
      "Number of transformer blocks."
    )
  }),
  dataset: Config.all({
    pretrainingPath: nonEmptyStringConfig(
      "EFFECT_GPT_PRETRAINING_PATH",
      "data/pretraining_data.json",
      "Path to the pretraining dataset file."
    ),
    chatPath: nonEmptyStringConfig(
      "EFFECT_GPT_CHAT_PATH",
      "data/chat_training_data.json",
      "Path to the chat fine-tuning dataset file."
    ),
    format: Config.literal("json", "csv")("EFFECT_GPT_DATASET_FORMAT").pipe(
      Config.withDefault("json"),
      Config.withDescription("Dataset format: json or csv.")
    )
  }),
  training: Config.all({
    pretraining: Config.all({
      epochs: positiveIntegerConfig("EFFECT_GPT_PRETRAIN_EPOCHS", 100, "Pretraining epochs."),
      learningRate: positiveNumberConfig("EFFECT_GPT_PRETRAIN_LR", 0.0005, "Pretraining learning rate.")
    }),
    finetuning: Config.all({
      epochs: positiveIntegerConfig("EFFECT_GPT_FINETUNE_EPOCHS", 100, "Instruction-tuning epochs."),
      learningRate: positiveNumberConfig("EFFECT_GPT_FINETUNE_LR", 0.0001, "Instruction-tuning learning rate.")
    })
  })
})

export const AppConfigLive = Layer.effect(AppConfig, appConfig)
