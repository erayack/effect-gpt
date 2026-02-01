import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { expectShape, expectNotClose, expectFinite } from "./support/tensorMatchers"
import { makeEmbeddings } from "./support/factories"
import * as T from "../../src/tensor/Tensor2D"
import { EMBEDDING_DIM, MAX_SEQ_LEN } from "../../src/config"

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

  test("parametersCount", () => {
    const vocabSize = 10
    const embeddings = makeEmbeddings(vocabSize)
    const expectedCount = vocabSize * EMBEDDING_DIM + MAX_SEQ_LEN * EMBEDDING_DIM
    expect(embeddings.parametersCount).toBe(expectedCount)
  })
})
