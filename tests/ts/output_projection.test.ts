import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeOutputProjection } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM } from "../../src/config"

describe("OutputProjection", () => {
  const vocabSize = 10

  test("weight dimensions [EMBEDDING_DIM, vocabSize]", () => {
    const proj = makeOutputProjection(vocabSize)
    expectShape(proj.wOut, [EMBEDDING_DIM, vocabSize])
  })

  test("bias dimensions [1, vocabSize]", () => {
    const proj = makeOutputProjection(vocabSize)
    expectShape(proj.bOut, [1, vocabSize])
  })

  test("forward [seqLen, dim] → [seqLen, vocabSize]", () => {
    const proj = makeOutputProjection(vocabSize)
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(proj.forward(input))
    expectShape(output, [3, vocabSize])
  })

  test("forward across sequence lengths", () => {
    const proj = makeOutputProjection(vocabSize)
    for (let seqLen = 1; seqLen <= 4; seqLen++) {
      const input = T.ones(seqLen, EMBEDDING_DIM)
      const output = runEffect(proj.forward(input))
      expectShape(output, [seqLen, vocabSize])
    }
  })

  test("output contains finite values", () => {
    const proj = makeOutputProjection(vocabSize)
    const input = T.ones(3, EMBEDDING_DIM)
    const output = runEffect(proj.forward(input))
    expectFinite(output)
  })

  test("backward gradient shape [seqLen, EMBEDDING_DIM]", () => {
    const proj = makeOutputProjection(vocabSize)
    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(proj.forward(input))
    const dOut = T.ones(3, vocabSize)
    const grad = runEffect(proj.backward(dOut, 0.01))
    expectShape(grad, [3, EMBEDDING_DIM])
  })

  test("backward updates wOut", () => {
    const proj = makeOutputProjection(vocabSize)
    const wOutBefore = T.clone(proj.wOut)
    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(proj.forward(input))
    const dOut = T.ones(3, vocabSize)
    runEffect(proj.backward(dOut, 0.01))
    expectNotClose(proj.wOut, wOutBefore)
  })

  test("backward updates bOut", () => {
    const proj = makeOutputProjection(vocabSize)
    const bOutBefore = T.clone(proj.bOut)
    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(proj.forward(input))
    const dOut = T.ones(3, vocabSize)
    runEffect(proj.backward(dOut, 0.01))
    expectNotClose(proj.bOut, bOutBefore)
  })

  test("parametersCount", () => {
    const proj = makeOutputProjection(vocabSize)
    const expected = EMBEDDING_DIM * vocabSize + vocabSize
    expect(proj.parametersCount).toBe(expected)
  })
})
