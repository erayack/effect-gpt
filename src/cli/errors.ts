import * as Cause from "effect/Cause"
import type { TrainingError } from "../errors"

const formatUnknown = (err: unknown): string => {
  if (Cause.isCause(err)) return Cause.pretty(err)
  if (err instanceof Error) return err.stack ?? `${err.name}: ${err.message}`
  return String(err)
}

export const formatTrainingError = (err: TrainingError | unknown): string => {
  if (err && typeof err === "object" && "_tag" in err) {
    const tagged = err as { _tag: string }
    switch (tagged._tag) {
      case "TrainingDatasetError": {
        const e = err as TrainingError & { cause: { path?: string; error?: unknown; message?: string; _tag?: string } }
        const location = e.cause.path ? ` (${e.cause.path})` : ""
        const reason = e.cause._tag ?? "dataset"
        const detail =
          e.cause.error !== undefined
            ? typeof e.cause.error === "object" && e.cause.error !== null && "message" in e.cause.error
              ? (e.cause.error as any).message
              : String(e.cause.error)
            : e.cause.message ?? ""
        return `Dataset error${location}: ${reason}${detail ? ` - ${detail}` : ""}`
      }
      case "TrainingShapeError": {
        const e = err as any
        return `Shape error: ${e.cause?.message ?? formatUnknown(e.cause)}`
      }
      case "TrainingTokenizerError": {
        const e = err as any
        return `Tokenizer error: ${e.message}`
      }
      case "TrainingOptimizerError": {
        const e = err as any
        return `Optimizer error: ${e.message}`
      }
      case "TrainingConfigError": {
        const e = err as any
        return `Configuration error: ${e.message}`
      }
      case "TrainingUnknownError": {
        const e = err as any
        return `Unexpected training error: ${formatUnknown(e.cause)}`
      }
      default:
        return `Unexpected error (${tagged._tag}): ${formatUnknown(err)}`
    }
  }
  return `Unexpected error: ${formatUnknown(err)}`
}
