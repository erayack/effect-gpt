import { describe, test, expect } from "bun:test"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { seeded } from "../../src/tensor/random"

const directArgmaxRows = (t: T.Tensor2D): ReadonlyArray<number> => {
  const result: Array<number> = []
  const { data, rows, cols } = t

  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    let maxIdx = 0
    let maxVal = data[rowOffset]

    for (let j = 1; j < cols; j++) {
      const val = data[rowOffset + j]
      if (val > maxVal) {
        maxVal = val
        maxIdx = j
      }
    }

    result.push(maxIdx)
  }

  return result
}

const tensorFromRows = (rows: ReadonlyArray<ReadonlyArray<number>>): T.Tensor2D => {
  const rowCount = rows.length
  const colCount = rows[0]!.length
  const flat = rows.flat()
  return T.fromArray(rowCount, colCount, flat)
}

describe("argmax(softmax(row)) equivalence", () => {
  test("matches direct argmax on representative finite rows", () => {
    const logits = tensorFromRows([
      [0, 0, 0, 0],
      [1, 2, 3, 4],
      [4, 3, 2, 1],
      [-1000, 0, 1000, 999],
      [10000, 10001, 9999, 10001],
      [1e-6, -1e-6, 2e-6, 0],
      [42, -42, 42, -42]
    ])

    const viaSoftmax = Ops.argmaxRows(Ops.softmaxRows(logits))
    const direct = directArgmaxRows(logits)

    expect(viaSoftmax).toEqual(direct)
  })

  test("matches direct argmax for deterministic random finite logits", () => {
    const rng = seeded(1337)
    const rows = 32
    const cols = 19
    const values = new Float32Array(rows * cols)

    for (let i = 0; i < values.length; i++) {
      // bounded finite range to avoid non-finite edge cases in this equivalence check
      values[i] = (rng.next() * 2 - 1) * 1_000
    }

    const logits = T.make(rows, cols, values)
    const viaSoftmax = Ops.argmaxRows(Ops.softmaxRows(logits))
    const direct = directArgmaxRows(logits)

    expect(viaSoftmax).toEqual(direct)
  })
})
