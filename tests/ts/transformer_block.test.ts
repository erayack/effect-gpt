import { describe, test, expect } from "bun:test"
import * as Effect from "effect/Effect"
import { runEffect, runEffectFail } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite, expectClose } from "./support/tensorMatchers"
import { makeTransformerBlock } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
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

  test("prefill matches forward", () => {
    const block = makeTransformerBlock()
    const input = T.ones(3, EMBEDDING_DIM)
    const expected = runEffect(block.forward(input))

    const incremental = makeTransformerBlock()
    incremental.attention.wQ = T.clone(block.attention.wQ)
    incremental.attention.wK = T.clone(block.attention.wK)
    incremental.attention.wV = T.clone(block.attention.wV)
    incremental.feedForward.w1 = T.clone(block.feedForward.w1)
    incremental.feedForward.b1 = T.clone(block.feedForward.b1)
    incremental.feedForward.w2 = T.clone(block.feedForward.w2)
    incremental.feedForward.b2 = T.clone(block.feedForward.b2)
    incremental.norm1.gamma = T.clone(block.norm1.gamma)
    incremental.norm1.beta = T.clone(block.norm1.beta)
    incremental.norm2.gamma = T.clone(block.norm2.gamma)
    incremental.norm2.beta = T.clone(block.norm2.beta)

    const state = incremental.createDecodeState(8)
    const actual = runEffect(incremental.prefill(input, state))

    expectClose(actual, expected)
    expect(state.attention.length).toBe(input.rows)
  })

  test("decodeStep matches last row of full forward on extended sequence", () => {
    const block = makeTransformerBlock()
    const prefix = T.ones(2, EMBEDDING_DIM)
    const next = T.zeros(1, EMBEDDING_DIM)
    for (let i = 0; i < next.data.length; i++) {
      next.data[i] = (i % 11) * 0.03
    }

    const extended = T.zeros(3, EMBEDDING_DIM)
    extended.data.set(prefix.data)
    extended.data.set(next.data, prefix.data.length)

    const expectedFull = runEffect(block.forward(extended))
    const expectedLast = runEffect(Ops.rowAsMatrix(expectedFull, expectedFull.rows - 1))

    const incremental = makeTransformerBlock()
    incremental.attention.wQ = T.clone(block.attention.wQ)
    incremental.attention.wK = T.clone(block.attention.wK)
    incremental.attention.wV = T.clone(block.attention.wV)
    incremental.feedForward.w1 = T.clone(block.feedForward.w1)
    incremental.feedForward.b1 = T.clone(block.feedForward.b1)
    incremental.feedForward.w2 = T.clone(block.feedForward.w2)
    incremental.feedForward.b2 = T.clone(block.feedForward.b2)
    incremental.norm1.gamma = T.clone(block.norm1.gamma)
    incremental.norm1.beta = T.clone(block.norm1.beta)
    incremental.norm2.gamma = T.clone(block.norm2.gamma)
    incremental.norm2.beta = T.clone(block.norm2.beta)

    const state = incremental.createDecodeState(8)
    runEffect(incremental.prefill(prefix, state))
    const actual = runEffect(incremental.decodeStep(next, state))

    expectClose(actual, expectedLast)
    expect(state.attention.length).toBe(3)
  })

  test("prefill rejects reused decode state", () => {
    const block = makeTransformerBlock()
    const state = block.createDecodeState(8)
    runEffect(block.prefill(T.ones(2, EMBEDDING_DIM), state))

    const error = runEffectFail(block.prefill(T.ones(1, EMBEDDING_DIM), state))

    expect(error.message).toContain("KV cache must be empty before prefill")
  })

  test("concurrent forward/backward uses isolated caches per fiber", async () => {
    const block = makeTransformerBlock()

    const inputA = T.ones(2, EMBEDDING_DIM)
    const inputB = T.zeros(3, EMBEDDING_DIM)
    for (let i = 0; i < inputB.data.length; i++) {
      inputB.data[i] = (i % 9) * 0.07
    }

    const gradA = T.ones(2, EMBEDDING_DIM)
    const gradB = T.ones(3, EMBEDDING_DIM)

    const runStep = (input: T.Tensor2D, grad: T.Tensor2D) =>
      block.forward(input).pipe(Effect.flatMap(() => block.backward(grad, 0)))

    const [resultA, resultB] = await Effect.runPromise(
      Effect.all([runStep(inputA, gradA), runStep(inputB, gradB)], { concurrency: "unbounded" })
    )

    expectShape(resultA, [2, EMBEDDING_DIM])
    expectShape(resultB, [3, EMBEDDING_DIM])
    expectFinite(resultA)
    expectFinite(resultB)
  })
})
