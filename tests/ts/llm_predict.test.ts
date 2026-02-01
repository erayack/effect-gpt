import { describe, test, expect } from "bun:test"
import * as Option from "effect/Option"
import { runEffect } from "./support/runEffect"
import { makeLLM, makeLLMWithNetwork, makeEmbeddings, makeTransformerBlock } from "./support/factories"
import { StubOutputProjection } from "./support/stubs"
import { Vocab } from "../../src/vocab/Vocab"
import { CANONICAL_SEED } from "./support/seed"

describe("LLM Predict", () => {
  const vocabWords = Vocab.defaultWords()
  const eosTokenId = vocabWords.indexOf("</s>")

  test("predict stops at EOS token", () => {
    const vocab = Vocab.make(vocabWords)
    const stubOutput = new StubOutputProjection(vocabWords.length, eosTokenId, 2)

    const llm = makeLLMWithNetwork({
      vocabWords,
      network: [
        makeEmbeddings(vocabWords.length, { seed: CANONICAL_SEED }),
        makeTransformerBlock({ seed: CANONICAL_SEED }),
        stubOutput
      ]
    })

    const result = runEffect(llm.predict("hello"))
    expect(result).toContain("</s>")
  })

  test("predict output token count respects EOS", () => {
    const stubOutput = new StubOutputProjection(vocabWords.length, eosTokenId, 3)

    const llm = makeLLMWithNetwork({
      vocabWords,
      network: [
        makeEmbeddings(vocabWords.length, { seed: CANONICAL_SEED }),
        makeTransformerBlock({ seed: CANONICAL_SEED }),
        stubOutput
      ]
    })

    const tokens = runEffect(llm.forward("hello"))
    expect(tokens.length).toBeLessThanOrEqual(3)
  })

  test("predict decodes tokens correctly", () => {
    const stubOutput = new StubOutputProjection(vocabWords.length, eosTokenId, 1)

    const llm = makeLLMWithNetwork({
      vocabWords,
      network: [
        makeEmbeddings(vocabWords.length, { seed: CANONICAL_SEED }),
        makeTransformerBlock({ seed: CANONICAL_SEED }),
        stubOutput
      ]
    })

    const tokens = runEffect(llm.forward("hello"))
    for (const tokenId of tokens) {
      const decoded = llm.vocab.decode(tokenId)
      expect(Option.isSome(decoded)).toBe(true)
    }
  })

  test("predict with empty input returns empty string", () => {
    const llm = makeLLM({ vocabWords })
    const result = runEffect(llm.predict(""))
    expect(result).toBe("")
  })

  test("forward with empty input returns empty tokens", () => {
    const llm = makeLLM({ vocabWords })
    const tokens = runEffect(llm.forward(""))
    expect(tokens.length).toBe(0)
  })

  test("predict with seeded RNG is deterministic", () => {
    const createLLM = () => {
      const stubOutput = new StubOutputProjection(vocabWords.length, eosTokenId, 2)
      return makeLLMWithNetwork({
        vocabWords,
        network: [
          makeEmbeddings(vocabWords.length, { seed: CANONICAL_SEED }),
          makeTransformerBlock({ seed: CANONICAL_SEED }),
          stubOutput
        ]
      })
    }

    const llm1 = createLLM()
    const llm2 = createLLM()

    const result1 = runEffect(llm1.predict("hello"))
    const result2 = runEffect(llm2.predict("hello"))

    expect(result1).toBe(result2)
  })
})
