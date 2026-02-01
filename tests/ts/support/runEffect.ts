/**
 * Helpers for running Effect programs in Bun tests.
 */
import * as Effect from "effect/Effect"

/**
 * Runs an Effect synchronously and returns the result.
 * Throws if the effect fails or requires async execution.
 * Use for pure, synchronous effects in tests.
 */
export const runEffect = <A, E>(effect: Effect.Effect<A, E>): A => Effect.runSync(effect)



/**
 * Runs an Effect and expects it to fail with a specific error type.
 * Returns the error for further assertions.
 */
export const runEffectFail = <A, E>(effect: Effect.Effect<A, E>): E => {
  const result = Effect.runSyncExit(effect)
  if (result._tag === "Failure") {
    const cause = result.cause
    if (cause._tag === "Fail") {
      return cause.error
    }
    throw new Error(`Effect failed with unexpected cause: ${cause._tag}`)
  }
  throw new Error("Expected Effect to fail, but it succeeded")
}
