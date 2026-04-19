import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { crossEntropyLossAndDLogits, crossEntropyLossAndDLogitsFromLogits } from "../../src/training/loss"
import { expectClose } from "./support/tensorMatchers"

describe("buffered tensor ops", () => {
  test("matMulInto matches allocating matMul", () => {
    const a = T.fromArray(2, 3, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(3, 2, [7, 8, 9, 10, 11, 12])

    const expected = runEffect(Ops.matMul(a, b))
    const actual = T.zeros(2, 2)
    runEffect(Ops.matMulInto(a, b, actual))

    expectClose(actual, expected)
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
