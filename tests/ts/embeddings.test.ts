import { describe, test, expect } from "bun:test"
import { runEffect, runEffectFail } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeEmbeddings } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM, MAX_SEQ_LEN } from "../../src/config"
import type { SequenceLayout } from "../../src/model/ModelLayer"

describe("Embeddings", () => {
  test("embed single token → [1, EMBEDDING_DIM]", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 1, [0])
    const output = runEffect(embeddings.forward(input))
    expectShape(output, [1, EMBEDDING_DIM])
    expectFinite(output)
  })

  test("embed multiple tokens → [seqLen, EMBEDDING_DIM]", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 3, [0, 1, 2])
    const output = runEffect(embeddings.forward(input))
    expectShape(output, [3, EMBEDDING_DIM])
    expectFinite(output)
  })

  test("positional embeddings differ across positions", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 3, [0, 0, 0])
    const output = runEffect(embeddings.forward(input))
    expectShape(output, [3, EMBEDDING_DIM])

    const row0Val = T.get(output, 0, 0)
    const row1Val = T.get(output, 1, 0)
    const row2Val = T.get(output, 2, 0)

    expect(row0Val).not.toBe(row1Val)
    expect(row1Val).not.toBe(row2Val)
    expect(row0Val).not.toBe(row2Val)
  })

  test("shape across sequence lengths 1..5", () => {
    const embeddings = makeEmbeddings(10)
    for (let seqLen = 1; seqLen <= 4; seqLen++) {
      const tokens = Array.from({ length: seqLen }, (_, i) => i % 10)
      const input = T.fromArray(1, seqLen, tokens)
      const output = runEffect(embeddings.forward(input))
      expectShape(output, [seqLen, EMBEDDING_DIM])
      expectFinite(output)
    }
  })

  test("max sequence length boundary", () => {
    const embeddings = makeEmbeddings(10)
    const tokens = Array.from({ length: MAX_SEQ_LEN }, (_, i) => i % 10)
    const input = T.fromArray(1, MAX_SEQ_LEN, tokens)
    const output = runEffect(embeddings.forward(input))
    expectShape(output, [MAX_SEQ_LEN, EMBEDDING_DIM])
    expectFinite(output)
  })

  test("forward rejects out-of-range token ids at the boundary", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 1, [10])

    const error = runEffectFail(embeddings.forward(input))

    expect(error.message).toContain("Token ID 10 out of bounds")
  })

  test("forward rejects out-of-range layout position ids at the boundary", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 1, [0])
    const layout: SequenceLayout = {
      totalTokens: 1,
      sequenceLengths: [1],
      sequenceIds: new Int32Array([0]),
      positionIds: new Int32Array([MAX_SEQ_LEN])
    }

    const error = runEffectFail(embeddings.forward(input, { sequenceLayout: layout }))

    expect(error.message).toContain(`Position ID ${MAX_SEQ_LEN} out of bounds`)
  })

  test("forwardTokenSync rejects out-of-range positions at the boundary", () => {
    const embeddings = makeEmbeddings(10)

    expect(() => embeddings.forwardTokenSync(0, MAX_SEQ_LEN)).toThrow("out of bounds")
  })

  test("backward updates token & positional embeddings", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 3, [0, 1, 2])

    const tokenBefore = T.clone(embeddings.tokenEmbeddings)
    const positionalBefore = T.clone(embeddings.positionalEmbeddings)

    runEffect(embeddings.forward(input))
    const grad = T.ones(3, EMBEDDING_DIM)
    runEffect(embeddings.backward(grad, 0.01))

    expectNotClose(embeddings.tokenEmbeddings, tokenBefore)
    expectNotClose(embeddings.positionalEmbeddings, positionalBefore)
  })

  test("batch layout resets positional embeddings per sequence", () => {
    const embeddings = makeEmbeddings(10)
    const flattenedInput = T.fromArray(1, 4, [0, 1, 2, 3])
    const layout: SequenceLayout = {
      totalTokens: 4,
      sequenceLengths: [2, 2],
      sequenceIds: new Int32Array([0, 0, 1, 1]),
      positionIds: new Int32Array([0, 1, 0, 1])
    }

    const batched = runEffect(embeddings.forward(flattenedInput, { sequenceLayout: layout }))
    const first = runEffect(embeddings.forward(T.fromArray(1, 2, [0, 1])))
    const second = runEffect(embeddings.forward(T.fromArray(1, 2, [2, 3])))

    expectShape(batched, [4, EMBEDDING_DIM])
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < EMBEDDING_DIM; col++) {
        expect(T.get(batched, row, col)).toBe(T.get(first, row, col))
        expect(T.get(batched, row + 2, col)).toBe(T.get(second, row, col))
      }
    }
  })

  test("batched backward accumulates positional gradients by positionIds", () => {
    const embeddings = makeEmbeddings(10)
    const input = T.fromArray(1, 4, [0, 1, 2, 3])
    const layout: SequenceLayout = {
      totalTokens: 4,
      sequenceLengths: [2, 2],
      sequenceIds: new Int32Array([0, 0, 1, 1]),
      positionIds: new Int32Array([0, 1, 0, 1])
    }
    const positionalBefore = T.clone(embeddings.positionalEmbeddings)

    runEffect(embeddings.forward(input, { sequenceLayout: layout }))
    runEffect(embeddings.backward(T.ones(4, EMBEDDING_DIM), 0.01))

    const rowDiff = (row: number) => {
      let diff = 0
      for (let col = 0; col < EMBEDDING_DIM; col++) {
        diff += Math.abs(
          T.get(positionalBefore, row, col) - T.get(embeddings.positionalEmbeddings, row, col)
        )
      }
      return diff
    }

    expect(rowDiff(0)).toBeGreaterThan(0)
    expect(rowDiff(1)).toBeGreaterThan(0)
    expect(rowDiff(2)).toBe(0)
    expect(rowDiff(3)).toBe(0)
  })

  test("parametersCount", () => {
    const vocabSize = 10
    const embeddings = makeEmbeddings(vocabSize)
    const expectedCount = vocabSize * EMBEDDING_DIM + MAX_SEQ_LEN * EMBEDDING_DIM
    expect(embeddings.parametersCount).toBe(expectedCount)
  })
})
