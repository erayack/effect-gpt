import { describe, test, expect } from "bun:test"
import { runEffect, runEffectFail } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite, expectClose } from "./support/tensorMatchers"
import { makeSelfAttention } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { EMBEDDING_DIM } from "../../src/config"
import type { SequenceLayout } from "../../src/model/ModelLayer"

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

  test("backward updates fused QKV projection weights", () => {
    const attention = makeSelfAttention()
    const wQKVBefore = T.clone(attention.wQKV)

    const input = T.ones(3, EMBEDDING_DIM)
    runEffect(attention.forward(input))
    const gradOut = T.ones(3, EMBEDDING_DIM)
    runEffect(attention.backward(gradOut, 0.01))

    expectNotClose(attention.wQKV, wQKVBefore)
  })

  test("batch layout prevents attention across flattened sequences", () => {
    const attention = makeSelfAttention()
    const sequenceA = T.fromArray(2, EMBEDDING_DIM, new Float32Array(2 * EMBEDDING_DIM).fill(1))
    const sequenceBData = new Float32Array(2 * EMBEDDING_DIM)
    sequenceBData.fill(5)
    const sequenceB = T.fromArray(2, EMBEDDING_DIM, sequenceBData)

    const flattenedData = new Float32Array(4 * EMBEDDING_DIM)
    flattenedData.set(sequenceA.data, 0)
    flattenedData.set(sequenceB.data, sequenceA.data.length)
    const flattened = T.make(4, EMBEDDING_DIM, flattenedData)

    const layout: SequenceLayout = {
      totalTokens: 4,
      sequenceLengths: [2, 2],
      sequenceIds: new Int32Array([0, 0, 1, 1]),
      positionIds: new Int32Array([0, 1, 0, 1])
    }

    const batched = runEffect(attention.forward(flattened, { sequenceLayout: layout }))
    const outputA = runEffect(attention.forward(sequenceA))
    const outputB = runEffect(attention.forward(sequenceB))

    expectShape(batched, [4, EMBEDDING_DIM])
    for (let col = 0; col < EMBEDDING_DIM; col++) {
      expect(T.get(batched, 0, col)).toBeCloseTo(T.get(outputA, 0, col), 5)
      expect(T.get(batched, 1, col)).toBeCloseTo(T.get(outputA, 1, col), 5)
      expect(T.get(batched, 2, col)).toBeCloseTo(T.get(outputB, 0, col), 5)
      expect(T.get(batched, 3, col)).toBeCloseTo(T.get(outputB, 1, col), 5)
    }
  })

  test("parametersCount", () => {
    const attention = makeSelfAttention()
    const expectedCount = 3 * EMBEDDING_DIM * EMBEDDING_DIM
    expect(attention.parametersCount).toBe(expectedCount)
  })

  test("prefill matches forward and seeds KV cache", () => {
    const attention = makeSelfAttention()
    const input = T.ones(3, EMBEDDING_DIM)

    const expected = runEffect(attention.forward(input))

    const cacheAttention = makeSelfAttention()
    cacheAttention.wQKV = T.clone(attention.wQKV)
    const cache = cacheAttention.createKvCache(8)

    const actual = runEffect(cacheAttention.prefill(input, cache))

    expectClose(actual, expected)
    expect(cache.length).toBe(input.rows)
  })

  test("decodeStep matches last row of full forward on extended sequence", () => {
    const attention = makeSelfAttention()
    const prefix = T.ones(3, EMBEDDING_DIM)
    const next = T.zeros(1, EMBEDDING_DIM)
    for (let i = 0; i < next.data.length; i++) {
      next.data[i] = (i % 7) * 0.05
    }

    const extended = T.zeros(4, EMBEDDING_DIM)
    extended.data.set(prefix.data)
    extended.data.set(next.data, prefix.data.length)

    const expectedFull = runEffect(attention.forward(extended))
    const expectedLast = runEffect(Ops.rowAsMatrix(expectedFull, expectedFull.rows - 1))

    const cacheAttention = makeSelfAttention()
    cacheAttention.wQKV = T.clone(attention.wQKV)
    const cache = cacheAttention.createKvCache(8)

    runEffect(cacheAttention.prefill(prefix, cache))
    const actual = runEffect(cacheAttention.decodeStep(next, cache))

    expectClose(actual, expectedLast)
    expect(cache.length).toBe(4)
  })

  test("prefill rejects non-empty KV cache", () => {
    const attention = makeSelfAttention()
    const cache = attention.createKvCache(8)
    runEffect(attention.prefill(T.ones(2, EMBEDDING_DIM), cache))

    const error = runEffectFail(attention.prefill(T.ones(1, EMBEDDING_DIM), cache))

    expect(error.message).toContain("KV cache must be empty before prefill")
  })
})
