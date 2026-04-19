import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeLayerNorm } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM } from "../../src/config"

const rowMean = (t: T.Tensor2D, row: number): number => {
  let sum = 0
  for (let j = 0; j < t.cols; j++) sum += T.get(t, row, j)
  return sum / t.cols
}

const rowVariance = (t: T.Tensor2D, row: number): number => {
  const mean = rowMean(t, row)
  let sumSq = 0
  for (let j = 0; j < t.cols; j++) {
    const diff = T.get(t, row, j) - mean
    sumSq += diff * diff
  }
  return sumSq / t.cols
}

const makeVariedInput = (): T.Tensor2D => {
  const input = T.zeros(3, EMBEDDING_DIM)
  for (let i = 0; i < input.data.length; i++) {
    input.data[i] = (i % 7) * 0.5 - 1.5
  }
  return input
}

describe("LayerNorm", () => {
  test("forward shape preserved", () => {
    const ln = makeLayerNorm()
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(ln.forward(input))
    expectShape(output, [3, EMBEDDING_DIM])
  })

  test("per-row mean approximately zero", () => {
    const ln = makeLayerNorm()
    const input = makeVariedInput()
    const output = runEffect(ln.forward(input))

    for (let row = 0; row < output.rows; row++) {
      const mean = rowMean(output, row)
      expect(Math.abs(mean)).toBeLessThan(1e-5)
    }
  })

  test("per-row variance approximately one", () => {
    const ln = makeLayerNorm()
    const input = makeVariedInput()
    const output = runEffect(ln.forward(input))

    for (let row = 0; row < output.rows; row++) {
      const variance = rowVariance(output, row)
      expect(Math.abs(variance - 1)).toBeLessThan(1e-4)
    }
  })

  test("output contains finite values", () => {
    const ln = makeLayerNorm()
    const input = makeVariedInput()
    const output = runEffect(ln.forward(input))
    expectFinite(output)
  })

  test("backward shape preserved", () => {
    const ln = makeLayerNorm()
    const input = makeVariedInput()
    runEffect(ln.forward(input))
    const dOut = T.ones(3, EMBEDDING_DIM)
    const grad = runEffect(ln.backward(dOut, 0.01))
    expectShape(grad, [3, EMBEDDING_DIM])
  })

  test("backward updates gamma and beta", () => {
    const ln = makeLayerNorm()
    const input = makeVariedInput()
    const gammaBefore = T.clone(ln.gamma)
    const betaBefore = T.clone(ln.beta)

    runEffect(ln.forward(input))
    const dOut = T.ones(3, EMBEDDING_DIM)
    runEffect(ln.backward(dOut, 0.01))

    expectNotClose(ln.gamma, gammaBefore)
    expectNotClose(ln.beta, betaBefore)
  })

  test("backward remains finite for zero-variance rows", () => {
    const ln = makeLayerNorm()
    const input = T.ones(3, EMBEDDING_DIM)

    runEffect(ln.forward(input))
    const grad = runEffect(ln.backward(T.ones(3, EMBEDDING_DIM), 0.01))

    expectShape(grad, [3, EMBEDDING_DIM])
    expectFinite(grad)
  })

  test("parametersCount", () => {
    const ln = makeLayerNorm()
    expect(ln.parametersCount).toBe(2 * EMBEDDING_DIM)
  })
})
