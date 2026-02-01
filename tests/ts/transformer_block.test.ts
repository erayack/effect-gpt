import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeTransformerBlock } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../../src/config"

describe("TransformerBlock", () => {
  test("forward shape preserved [1, EMBEDDING_DIM]", () => {
    const block = makeTransformerBlock()
    const input = T.ones(1, EMBEDDING_DIM)
    const output = runEffect(block.forward(input))
    expectShape(output, [1, EMBEDDING_DIM])
  })

  test("forward shape preserved across sequence lengths", () => {
    const block = makeTransformerBlock()
    for (let seqLen = 1; seqLen <= 4; seqLen++) {
      const input = T.ones(seqLen, EMBEDDING_DIM)
      const output = runEffect(block.forward(input))
      expectShape(output, [seqLen, EMBEDDING_DIM])
    }
  })

  test("output contains finite values", () => {
    const block = makeTransformerBlock()
    const input = T.ones(2, EMBEDDING_DIM)
    const output = runEffect(block.forward(input))
    expectFinite(output)
  })

  test("backward shape matches input", () => {
    const block = makeTransformerBlock()
    const input = T.ones(2, EMBEDDING_DIM)
    runEffect(block.forward(input))
    const dOut = T.ones(2, EMBEDDING_DIM)
    const grad = runEffect(block.backward(dOut, 0.01))
    expectShape(grad, [2, EMBEDDING_DIM])
  })

  test("backward updates sub-layer weights", () => {
    const block = makeTransformerBlock()
    const input = T.zeros(2, EMBEDDING_DIM)
    for (let i = 0; i < input.data.length; i++) {
      input.data[i] = (i % 10) * 0.1
    }

    const wQBefore = T.clone(block.attention.wQ)
    const w1Before = T.clone(block.feedForward.w1)
    const betaBefore = T.clone(block.norm1.beta)

    runEffect(block.forward(input))
    const dOut = T.ones(2, EMBEDDING_DIM)
    runEffect(block.backward(dOut, 0.01))

    expectNotClose(block.attention.wQ, wQBefore)
    expectNotClose(block.feedForward.w1, w1Before)
    expectNotClose(block.norm1.beta, betaBefore)
  })

  test("parametersCount equals sum of components", () => {
    const block = makeTransformerBlock()

    const attentionParams = 3 * EMBEDDING_DIM * EMBEDDING_DIM
    const feedForwardParams =
      EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM
    const normParams = 2 * (2 * EMBEDDING_DIM)

    const expectedTotal = attentionParams + feedForwardParams + normParams
    expect(block.parametersCount).toBe(expectedTotal)
  })
})
