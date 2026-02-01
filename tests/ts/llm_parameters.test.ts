import { describe, test, expect } from "bun:test"
import { LLM } from "../../src/model/LLM"
import { Vocab } from "../../src/vocab/Vocab"
import { EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN } from "../../src/config"
import { makeLLM } from "./support/factories"
import { seeded } from "../../src/tensor/random"

describe("LLM Parameters", () => {
  const computeExpectedParams = (vocabSize: number, numTransformerBlocks: number): number => {
    const embeddingsParams = vocabSize * EMBEDDING_DIM + MAX_SEQ_LEN * EMBEDDING_DIM

    const selfAttentionParams = 3 * EMBEDDING_DIM * EMBEDDING_DIM
    const feedForwardParams =
      EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM
    const layerNormParams = 2 * (2 * EMBEDDING_DIM)
    const transformerBlockParams = selfAttentionParams + feedForwardParams + layerNormParams

    const outputProjectionParams = EMBEDDING_DIM * vocabSize + vocabSize

    return embeddingsParams + numTransformerBlocks * transformerBlockParams + outputProjectionParams
  }

  test("default LLM parameter count matches formula", () => {
    const llm = LLM.default(seeded(1))
    const vocabSize = llm.vocab.words.length
    const expected = computeExpectedParams(vocabSize, 1)
    expect(llm.totalParameters()).toBe(expected)
  })

  test("make LLM parameter count matches formula (3 transformer blocks)", () => {
    const vocab = Vocab.make(Vocab.defaultWords())
    const llm = LLM.make(vocab, seeded(2))
    const vocabSize = vocab.words.length
    const expected = computeExpectedParams(vocabSize, 3)
    expect(llm.totalParameters()).toBe(expected)
  })

  test("factory LLM with 1 block matches formula", () => {
    const vocabWords = Vocab.defaultWords()
    const llm = makeLLM({ vocabWords, numTransformerBlocks: 1 })
    const expected = computeExpectedParams(vocabWords.length, 1)
    expect(llm.totalParameters()).toBe(expected)
  })

  test("factory LLM with 2 blocks matches formula", () => {
    const vocabWords = Vocab.defaultWords()
    const llm = makeLLM({ vocabWords, numTransformerBlocks: 2 })
    const expected = computeExpectedParams(vocabWords.length, 2)
    expect(llm.totalParameters()).toBe(expected)
  })

  test("individual layer params sum to total", () => {
    const llm = LLM.default(seeded(3))
    const sumOfLayers = llm.network.reduce((sum, layer) => sum + layer.parametersCount, 0)
    expect(llm.totalParameters()).toBe(sumOfLayers)
  })

  test("larger vocab increases parameter count", () => {
    const smallVocab = ["a", "b", "</s>"]
    const largeVocab = ["a", "b", "c", "d", "e", "f", "g", "h", "</s>"]

    const smallLLM = makeLLM({ vocabWords: smallVocab, numTransformerBlocks: 1 })
    const largeLLM = makeLLM({ vocabWords: largeVocab, numTransformerBlocks: 1 })

    expect(largeLLM.totalParameters()).toBeGreaterThan(smallLLM.totalParameters())
  })

  test("more transformer blocks increases parameter count", () => {
    const vocabWords = Vocab.defaultWords()
    const llm1Block = makeLLM({ vocabWords, numTransformerBlocks: 1 })
    const llm3Blocks = makeLLM({ vocabWords, numTransformerBlocks: 3 })

    expect(llm3Blocks.totalParameters()).toBeGreaterThan(llm1Block.totalParameters())
  })

  test("parameter count formula components", () => {
    const vocabSize = 6
    const numBlocks = 1

    const embeddingsParams = vocabSize * EMBEDDING_DIM + MAX_SEQ_LEN * EMBEDDING_DIM
    expect(embeddingsParams).toBe(6 * 128 + 80 * 128)

    const selfAttentionParams = 3 * EMBEDDING_DIM * EMBEDDING_DIM
    expect(selfAttentionParams).toBe(3 * 128 * 128)

    const feedForwardParams =
      EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM
    expect(feedForwardParams).toBe(128 * 256 + 256 + 256 * 128 + 128)

    const layerNormParams = 2 * (2 * EMBEDDING_DIM)
    expect(layerNormParams).toBe(2 * (2 * 128))

    const outputProjectionParams = EMBEDDING_DIM * vocabSize + vocabSize
    expect(outputProjectionParams).toBe(128 * 6 + 6)
  })
})
