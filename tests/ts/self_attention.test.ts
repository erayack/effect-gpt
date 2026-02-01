import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeSelfAttention } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM } from "../../src/config"

describe("SelfAttention", () => {
  test("forward shape matches input", () => {
    const attention = makeSelfAttention()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(attention.forward(input))
    expectShape(output, [3, EMBEDDING_DIM])
  })

  test("shape across sequence lengths 1..5", () => {
    const attention = makeSelfAttention()
    for (let seqLen = 1; seqLen <= 4; seqLen++) {
      const input = T.ones(seqLen, EMBEDDING_DIM)
      const output = runEffect(attention.forward(input))
      expectShape(output, [seqLen, EMBEDDING_DIM])
    }
  })

  test("output contains finite values", () => {
    const attention = makeSelfAttention()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(attention.forward(input))
    expectFinite(output)
  })

  test("backward returns correct shape", () => {
    const attention = makeSelfAttention()
    const seqLen = 3
    const input = T.ones(seqLen, EMBEDDING_DIM)
    runEffect(attention.forward(input))
    const gradOut = T.ones(seqLen, EMBEDDING_DIM)
    const grad = runEffect(attention.backward(gradOut, 0.01))
    expectShape(grad, [seqLen, EMBEDDING_DIM])
  })

  test("backward updates wQ/wK/wV weights", () => {
    const attention = makeSelfAttention()
    const wQBefore = T.clone(attention.wQ)
    const wKBefore = T.clone(attention.wK)
    const wVBefore = T.clone(attention.wV)

    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(attention.forward(input))
    const gradOut = T.ones(3, EMBEDDING_DIM)
    runEffect(attention.backward(gradOut, 0.01))

    expectNotClose(attention.wQ, wQBefore)
    expectNotClose(attention.wK, wKBefore)
    expectNotClose(attention.wV, wVBefore)
  })

  test("parametersCount", () => {
    const attention = makeSelfAttention()
    const expectedCount = 3 * EMBEDDING_DIM * EMBEDDING_DIM
    expect(attention.parametersCount).toBe(expectedCount)
  })
})
