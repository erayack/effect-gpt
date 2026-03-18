import * as Effect from "effect/Effect"
import * as Cause from "effect/Cause"
import { TrainingError } from "../errors"
import type { TrainingError as TrainingErrorType } from "../errors"

const formatUnknown = (err: unknown): string => {
  if (Cause.isCause(err)) return Cause.pretty(err)
  if (err instanceof Error) return err.stack ?? `${err.name}: ${err.message}`
  return String(err)
}

const formatKnownTrainingError = (error: TrainingErrorType): Effect.Effect<string> =>
  Effect.fail(error).pipe(
    Effect.catchTag("TrainingDatasetError", (e) => {
      const location = e.cause.path ? ` (${e.cause.path})` : ""
      const reason = e.cause._tag ?? "dataset"
      const detail =
        e.cause.error !== undefined
          ? typeof e.cause.error === "object" && e.cause.error !== null && "message" in e.cause.error
            ? (e.cause.error as { message: string }).message
            : String(e.cause.error)
          : "message" in e.cause && typeof e.cause.message === "string"
            ? e.cause.message
            : ""
      return Effect.succeed(`Dataset error${location}: ${reason}${detail ? ` - ${detail}` : ""}`)
    }),
    Effect.catchTag("TrainingShapeError", (e) =>
      Effect.succeed(`Shape error: ${e.cause?.message ?? formatUnknown(e.cause)}`)
    ),
    Effect.catchTag("TrainingTokenizerError", (e) =>
      Effect.succeed(`Tokenizer error: ${e.message}`)
    ),
    Effect.catchTag("TrainingOptimizerError", (e) =>
      Effect.succeed(`Optimizer error: ${e.message}`)
    ),
    Effect.catchTag("TrainingConfigError", (e) =>
      Effect.succeed(`Configuration error: ${e.message}`)
    ),
    Effect.catchTag("TrainingUnknownError", (e) =>
      Effect.succeed(`Unexpected training error: ${formatUnknown(e.cause)}`)
    )
  )

export const formatTrainingError = (error: TrainingErrorType | unknown): string =>
  Effect.runSync(formatKnownTrainingError(TrainingError.fromUnknown(error)))
