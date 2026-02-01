import { describe, expect, test } from "bun:test"
import { TrainingError } from "../../src/errors"
import { ShapeError } from "../../src/tensor/ops"
import { DatasetLoadError, DatasetParseError } from "../../src/data/Dataset"

describe("TrainingError.fromUnknown", () => {
  test("maps ShapeError to TrainingShapeError", () => {
    const err = TrainingError.fromUnknown(new ShapeError("bad shape"))
    expect(err._tag).toBe("TrainingShapeError")
    expect((err as any).cause.message).toBe("bad shape")
  })

  test("maps DatasetLoadError to TrainingDatasetError", () => {
    const loadErr = new DatasetLoadError({ path: "x", error: new Error("io") })
    const err = TrainingError.fromUnknown(loadErr)
    expect(err._tag).toBe("TrainingDatasetError")
    expect((err as any).cause).toBe(loadErr)
  })

  test("maps DatasetParseError to TrainingDatasetError", () => {
    const parseErr = new DatasetParseError({ path: "x", error: new Error("parse") })
    const err = TrainingError.fromUnknown(parseErr)
    expect(err._tag).toBe("TrainingDatasetError")
  })

  test("passes through existing TrainingError", () => {
    const existing = TrainingError.optimizer("boom")
    const err = TrainingError.fromUnknown(existing)
    expect(err).toBe(existing)
  })

  test("wraps unknown errors in TrainingUnknownError", () => {
    const err = TrainingError.fromUnknown(new Error("boom"))
    expect(err._tag).toBe("TrainingUnknownError")
  })
})
