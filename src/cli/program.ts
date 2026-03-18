import * as Effect from "effect/Effect"
import type { LoggerServiceId } from "../services/Logger"
import { error as logError } from "../services/Logger"
import { formatTrainingError } from "./errors"

export const withCliErrorLogging = <A, E, R>(
  effect: Effect.Effect<A, E, R>
): Effect.Effect<A, E, R | LoggerServiceId> =>
  effect.pipe(
    Effect.tapError((error) =>
      logError(formatTrainingError(error))
    )
  )
