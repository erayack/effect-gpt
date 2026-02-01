import { describe, test, expect } from "bun:test"
import * as Effect from "effect/Effect"
import * as Exit from "effect/Exit"
import { BunFileSystem } from "@effect/platform-bun"
import { Dataset, DatasetParseError } from "../../src/data/Dataset.js"

describe("Dataset CSV", () => {
  const good = "tests/fixtures/csv_good.csv"
  const bad = "tests/fixtures/csv_bad.csv"

  test("collect reads CSV and joins fields", async () => {
    const streams = Dataset.load({ pretrainingPath: good, chatPath: good, format: "csv" })
    const program = Dataset.collect(streams).pipe(Effect.provide(BunFileSystem.layer))
    const result = await Effect.runPromise(program)

    expect(result.pretrainingData).toEqual(["hello,world", "quoted,field,foo", 'say "hi",bar'])
    expect(result.chatTrainingData).toEqual(["hello,world", "quoted,field,foo", 'say "hi",bar'])
  })

  test("malformed CSV surfaces DatasetParseError", async () => {
    const streams = Dataset.load({ pretrainingPath: bad, chatPath: bad, format: "csv" })
    const program = Dataset.collect(streams).pipe(Effect.provide(BunFileSystem.layer))
    const exit = await Effect.runPromiseExit(program)

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const cause = exit.cause
      if (cause._tag === "Fail") {
        expect(cause.error).toBeInstanceOf(DatasetParseError)
      } else {
        throw new Error(`Unexpected failure cause: ${cause._tag}`)
      }
    }
  })
})
