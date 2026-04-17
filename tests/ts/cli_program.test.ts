import { describe, expect, test } from "bun:test"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import { withCliErrorLogging } from "../../src/cli/program"
import { Logger } from "../../src/services/Logger"

describe("withCliErrorLogging", () => {
  test("preserves success result without logging", () => {
    const logs: Array<string> = []
    const loggerLayer = Layer.succeed(Logger, {
      log: () => Effect.void,
      debug: () => Effect.void,
      info: () => Effect.void,
      warn: () => Effect.void,
      error: (message: string, _data?: Record<string, unknown>) =>
        Effect.sync(() => {
          logs.push(message)
        })
    })

    const exit = Effect.runSyncExit(
      withCliErrorLogging(Effect.succeed(42)).pipe(Effect.provide(loggerLayer))
    )

    expect(exit._tag).toBe("Success")
    if (exit._tag === "Success") {
      expect(exit.value).toBe(42)
    }
    expect(logs).toEqual([])
  })

  test("logs formatted error and keeps failure", () => {
    const logs: Array<string> = []
    const loggerLayer = Layer.succeed(Logger, {
      log: () => Effect.void,
      debug: () => Effect.void,
      info: () => Effect.void,
      warn: () => Effect.void,
      error: (message: string, _data?: Record<string, unknown>) =>
        Effect.sync(() => {
          logs.push(message)
        })
    })

    const boom = new Error("boom")
    const exit = Effect.runSyncExit(
      withCliErrorLogging(Effect.fail(boom)).pipe(Effect.provide(loggerLayer))
    )

    expect(exit._tag).toBe("Failure")
    expect(logs).toHaveLength(1)
    expect(logs[0]).toContain("Unexpected training error")
    expect(logs[0]).toContain("boom")
  })
})
