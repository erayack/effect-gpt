import { describe, expect, test } from "bun:test"
import { formatTrainingError } from "../../src/cli/errors"
import { TrainingError } from "../../src/errors"
import { ShapeError } from "../../src/tensor/ops"

describe("formatTrainingError", () => {
  test("formats dataset error with path", () => {
    const err = TrainingError.dataset({ _tag: "DatasetLoadError", path: "/tmp/data", error: new Error("io") } as any)
    const message = formatTrainingError(err)
    expect(message).toContain("Dataset error")
    expect(message).toContain("/tmp/data")
  })

  test("formats shape error cause", () => {
    const err = TrainingError.shape(new ShapeError("bad shape"))
    const message = formatTrainingError(err)
    expect(message).toContain("Shape error")
    expect(message).toContain("bad shape")
  })

  test("formats plain Error as unexpected training error", () => {
    const message = formatTrainingError(new Error("boom"))
    expect(message).toContain("Unexpected training error")
    expect(message).toContain("boom")
  })
})
