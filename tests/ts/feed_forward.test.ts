import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeFeedForward } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../../src/config"

describe("FeedForward", () => {
  test("forward shape matches input", () => {
    const ff = makeFeedForward()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(ff.forward(input))
    expectShape(output, [3, EMBEDDING_DIM])
  })

  test("shape across sequence lengths 1..5", () => {
    const ff = makeFeedForward()
    for (let seqLen = 1; seqLen <= 4; seqLen++) {
      const input = T.ones(seqLen, EMBEDDING_DIM)
      const output = runEffect(ff.forward(input))
      expectShape(output, [seqLen, EMBEDDING_DIM])
    }
  })

  test("output contains finite values", () => {
    const ff = makeFeedForward()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(ff.forward(input))
    expectFinite(output)
  })

  test("backward returns gradient with correct shape", () => {
    const ff = makeFeedForward()
    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(ff.forward(input))
    const grads = T.ones(3, EMBEDDING_DIM)
    const gradInput = runEffect(ff.backward(grads, 0.01))
    expectShape(gradInput, [3, EMBEDDING_DIM])
  })

  test("backward output differs from forward output", () => {
    const ff = makeFeedForward()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(ff.forward(input))
    const grads = T.ones(3, EMBEDDING_DIM)
    const gradInput = runEffect(ff.backward(grads, 0.01))
    expectNotClose(output, gradInput)
  })

  test("backward updates w1/b1/w2/b2", () => {
    const ff = makeFeedForward()
    const w1Before = T.clone(ff.w1)
    const b1Before = T.clone(ff.b1)
    const w2Before = T.clone(ff.w2)
    const b2Before = T.clone(ff.b2)

    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(ff.forward(input))
    const grads = T.ones(3, EMBEDDING_DIM)
    runEffect(ff.backward(grads, 0.01))

    expectNotClose(ff.w1, w1Before)
    expectNotClose(ff.b1, b1Before)
    expectNotClose(ff.w2, w2Before)
    expectNotClose(ff.b2, b2Before)
  })

  test("backward remains finite when ReLU outputs are zero", () => {
    const ff = makeFeedForward()
    ff.w1.data.fill(0)
    ff.b1.data.fill(-1)
    ff.w2.data.fill(0)
    ff.b2.data.fill(0)

    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(ff.forward(input))
    expectFinite(output)

    const gradInput = runEffect(ff.backward(T.ones(3, EMBEDDING_DIM), 0.01))
    expectShape(gradInput, [3, EMBEDDING_DIM])
    expectFinite(gradInput)
  })

  test("parametersCount", () => {
    const ff = makeFeedForward()
    const expected =
      EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM
    expect(ff.parametersCount).toBe(expected)
  })
})
