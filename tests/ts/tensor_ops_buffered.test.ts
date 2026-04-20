import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { runEffectFail } from "./support/runEffect"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { crossEntropyLossAndDLogits, crossEntropyLossAndDLogitsFromLogits } from "../../src/training/loss"
import { expectClose } from "./support/tensorMatchers"
import { TensorWorkspace } from "../../src/tensor/Workspace"

const makePatternedTensor = (rows: number, cols: number, offset: number): T.Tensor2D => {
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i + offset) % 17) / 5 - 1.5
  }
  return T.make(rows, cols, data)
}

const referenceMatMul = (
  a: T.Tensor2D,
  b: T.Tensor2D,
  options?: { transposeA?: boolean; transposeB?: boolean }
): T.Tensor2D => {
  const transposeA = options?.transposeA === true
  const transposeB = options?.transposeB === true
  const rows = transposeA ? a.cols : a.rows
  const sharedDim = transposeA ? a.rows : a.cols
  const cols = transposeB ? b.rows : b.cols
  const out = T.zeros(rows, cols)

  for (let row = 0; row < rows; row++) {
    const outRowOffset = row * cols
    for (let col = 0; col < cols; col++) {
      let sum = 0
      for (let k = 0; k < sharedDim; k++) {
        const aValue = transposeA ? a.data[k * a.cols + row]! : a.data[row * a.cols + k]!
        const bValue = transposeB ? b.data[col * b.cols + k]! : b.data[k * b.cols + col]!
        sum += aValue * bValue
      }
      out.data[outRowOffset + col] = sum
    }
  }

  return out
}

describe("buffered tensor ops", () => {
  test("matMulInto matches allocating matMul", () => {
    const a = T.fromArray(2, 3, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(3, 2, [7, 8, 9, 10, 11, 12])

    const expected = runEffect(Ops.matMul(a, b))
    const actual = T.zeros(2, 2)
    runEffect(Ops.matMulInto(a, b, actual))

    expectClose(actual, expected)
  })

  test("matMulInto supports transposeA", () => {
    const a = T.fromArray(3, 2, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(3, 2, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)

    runEffect(Ops.matMulInto(a, b, actual, { transposeA: true }))

    expectClose(actual, T.fromArray(2, 2, [89, 98, 116, 128]))
  })

  test("matMulInto supports transposeB with workspace reuse", () => {
    const a = T.fromArray(2, 3, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(2, 3, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)
    const workspace = new TensorWorkspace()

    runEffect(Ops.matMulInto(a, b, actual, { transposeB: true, workspace }))

    expectClose(actual, T.fromArray(2, 2, [50, 68, 122, 167]))
  })

  test("matMulInto supports transposeA and transposeB together", () => {
    const a = T.fromArray(3, 2, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(2, 3, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)

    runEffect(Ops.matMulInto(a, b, actual, { transposeA: true, transposeB: true }))

    expectClose(actual, T.fromArray(2, 2, [76, 103, 100, 136]))
  })

  test("matMulInto rejects aliasing output storage", () => {
    const a = T.fromArray(2, 2, [1, 2, 3, 4])
    const b = T.fromArray(2, 2, [5, 6, 7, 8])

    const error = runEffectFail(Ops.matMulInto(a, b, a))

    expect(error.message).toContain("must not alias input storage")
  })

  test("blocked matMulInto matches reference implementation across transpose modes", () => {
    const nnA = makePatternedTensor(24, 48, 1)
    const nnB = makePatternedTensor(48, 40, 2)
    const nnActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(nnA, nnB, nnActual))
    expectClose(nnActual, referenceMatMul(nnA, nnB))

    const ntA = makePatternedTensor(24, 48, 3)
    const ntB = makePatternedTensor(40, 48, 4)
    const ntActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(ntA, ntB, ntActual, { transposeB: true }))
    expectClose(ntActual, referenceMatMul(ntA, ntB, { transposeB: true }))

    const tnA = makePatternedTensor(48, 24, 5)
    const tnB = makePatternedTensor(48, 40, 6)
    const tnActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(tnA, tnB, tnActual, { transposeA: true }))
    expectClose(tnActual, referenceMatMul(tnA, tnB, { transposeA: true }))

    const ttA = makePatternedTensor(48, 24, 7)
    const ttB = makePatternedTensor(40, 48, 8)
    const ttActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(ttA, ttB, ttActual, { transposeA: true, transposeB: true }))
    expectClose(ttActual, referenceMatMul(ttA, ttB, { transposeA: true, transposeB: true }))
  })

  test("softmaxRowsInto matches allocating softmaxRows", () => {
    const logits = T.fromArray(2, 4, [1, 2, 3, 4, 4, 3, 2, 1])
    const expected = Ops.softmaxRows(logits)
    const actual = T.zeros(2, 4)

    Ops.softmaxRowsInto(logits, actual)

    expectClose(actual, expected)
  })

  test("softmaxRowsInPlace normalizes each row", () => {
    const logits = T.fromArray(2, 3, [1, 2, 3, -1, 0, 1])

    Ops.softmaxRowsInPlace(logits)

    for (let row = 0; row < logits.rows; row++) {
      const rowOffset = row * logits.cols
      let sum = 0
      for (let col = 0; col < logits.cols; col++) {
        sum += logits.data[rowOffset + col]!
      }
      expect(sum).toBeCloseTo(1, 6)
    }
  })

  test("gatherRowsInto matches allocating gatherRows", () => {
    const embeddings = T.fromArray(4, 2, [1, 10, 2, 20, 3, 30, 4, 40])
    const tokenIds = [3, 1, 0]

    const expected = runEffect(Ops.gatherRows(embeddings, tokenIds))
    const actual = T.zeros(3, 2)
    runEffect(Ops.gatherRowsInto(embeddings, tokenIds, actual))

    expectClose(actual, expected)
  })

  test("crossEntropyLossAndDLogitsFromLogits matches softmax-based path", () => {
    const logits = T.fromArray(3, 4, [1, 3, 2, 0, -1, 0, 2, 4, 5, 1, 0, -2])
    const targets = [1, 3, 0]

    const probs = Ops.softmaxRows(logits)
    const expected = crossEntropyLossAndDLogits(probs, targets)
    const actual = crossEntropyLossAndDLogitsFromLogits(logits, targets)

    expect(actual.loss).toBeCloseTo(expected.loss, 6)
    expectClose(actual.grads, expected.grads)
  })
})
