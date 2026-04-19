export { Vocab } from "./vocab/Vocab"
export { tokenize } from "./tokenize/tokenize"
export { Dataset, DatasetLoadError, DatasetParseError } from "./data/Dataset"
export {
  TrainingError,
  TrainingDatasetError,
  TrainingShapeError,
  TrainingTokenizerError,
  TrainingOptimizerError,
  TrainingConfigError,
  TrainingUnknownError
} from "./errors"
export * from "./config"

export type { Tensor2D } from "./tensor/Tensor2D"
export * as T2D from "./tensor/Tensor2D"
export * as TensorOps from "./tensor/ops"
export { ShapeError } from "./tensor/ops"
export type { Rng } from "./tensor/random"
export { seeded } from "./tensor/random"
export { systemRng } from "./tensor/random"

export type { LayerForwardContext, ModelLayer, SequenceLayout } from "./model/ModelLayer"
export { Embeddings } from "./model/Embeddings"
export { SelfAttention } from "./model/SelfAttention"
export type { SelfAttentionKvCache } from "./model/SelfAttention"
export { FeedForward } from "./model/FeedForward"
export { LayerNorm } from "./model/LayerNorm"
export { TransformerBlock } from "./model/TransformerBlock"
export type { TransformerBlockDecodeState } from "./model/TransformerBlock"
export { OutputProjection } from "./model/OutputProjection"
export { LLM } from "./model/LLM"

export { Adam } from "./training/Adam"
export { clipGlobalL2 } from "./training/clip"
export { softmaxRows, crossEntropyLoss, dLogits } from "./training/loss"
export {
  train,
  trainStream,
  LLMService,
  TrainingConfig,
  makeLLMLayer,
  makeTrainingConfigLayer,
  makePreprocessSettingsLayer,
  DefaultPreprocessSettings,
  DefaultPreprocessSettingsLive
} from "./training/train"

export type {
  LogLevel,
  LoggerService,
  LoggerServiceId,
  RandomService,
  RandomServiceId,
  SeedService,
  SeedServiceId,
  MetricsService,
  MetricsServiceId,
  MetricsSnapshot,
  Counter,
  Gauge,
  Histogram,
  TimingResult
} from "./services"
export {
  Logger,
  ConsoleLoggerLive,
  TerminalLoggerLive,
  NullLoggerLive,
  log,
  debug,
  info,
  warn,
  error,
  Random,
  SeededRandomLive,
  SystemRandomLive,
  next,
  nextGaussian,
  nextInt,
  fork,
  Seed,
  SeedLayer,
  useSeedRng,
  Metrics,
  InMemoryMetricsLive,
  NoOpMetricsLive,
  counter,
  gauge,
  histogram,
  timed,
  snapshot
} from "./services"
