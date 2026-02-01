import * as Data from "effect/Data"
import type { DatasetLoadError, DatasetParseError } from "./data/Dataset"
import type { ShapeError } from "./tensor/ops"

/**
 * Narrow, discriminated error channel for the training pipeline.
 * Keeps original domain errors attached for rich reporting at the CLI boundary.
 */
export class TrainingDatasetError extends Data.TaggedError("TrainingDatasetError")<{
  readonly cause: DatasetLoadError | DatasetParseError
}> {}

export class TrainingShapeError extends Data.TaggedError("TrainingShapeError")<{
  readonly cause: ShapeError
}> {}

export class TrainingTokenizerError extends Data.TaggedError("TrainingTokenizerError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

export class TrainingOptimizerError extends Data.TaggedError("TrainingOptimizerError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

export class TrainingConfigError extends Data.TaggedError("TrainingConfigError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

export class TrainingUnknownError extends Data.TaggedError("TrainingUnknownError")<{
  readonly cause: unknown
}> {}

export type TrainingError =
  | TrainingDatasetError
  | TrainingShapeError
  | TrainingTokenizerError
  | TrainingOptimizerError
  | TrainingConfigError
  | TrainingUnknownError

export const TrainingError = {
  dataset: (cause: DatasetLoadError | DatasetParseError): TrainingDatasetError =>
    new TrainingDatasetError({ cause }),
  shape: (cause: ShapeError): TrainingShapeError => new TrainingShapeError({ cause }),
  tokenizer: (message: string, cause?: unknown): TrainingTokenizerError =>
    new TrainingTokenizerError({ message, cause }),
  optimizer: (message: string, cause?: unknown): TrainingOptimizerError =>
    new TrainingOptimizerError({ message, cause }),
  config: (message: string, cause?: unknown): TrainingConfigError =>
    new TrainingConfigError({ message, cause }),
  unknown: (cause: unknown): TrainingUnknownError => new TrainingUnknownError({ cause }),
  fromUnknown: (error: unknown): TrainingError => {
    if (error instanceof TrainingDatasetError) return error
    if (error instanceof TrainingShapeError) return error
    if (error instanceof TrainingTokenizerError) return error
    if (error instanceof TrainingOptimizerError) return error
    if (error instanceof TrainingConfigError) return error
    // Upstream domain errors
    if (error && typeof error === "object") {
      const candidate = error as { _tag?: string }
      if (candidate._tag === "DatasetLoadError" || candidate._tag === "DatasetParseError") {
        return new TrainingDatasetError({ cause: error as DatasetLoadError | DatasetParseError })
      }
      if (candidate._tag === "ShapeError") {
        return new TrainingShapeError({ cause: error as ShapeError })
      }
    }
    return new TrainingUnknownError({ cause: error })
  }
}
