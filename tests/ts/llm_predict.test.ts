import { describe, test, expect } from "bun:test"
import * as Option from "effect/Option"
import { runEffect } from "./support/runEffect"
import { makeLLM, makeLLMWithNetwork, makeEmbeddings, makeTransformerBlock } from "./support/factories"
import { StubOutputProjection } from "./support/stubs"
import { Vocab } from "../../src/vocab/Vocab"
import { CANONICAL_SEED } from "./support/seed"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { MAX_SEQ_LEN } from "../../src/config"
import { tokenize } from "../../src/tokenize/tokenize"
import type { LLM } from "../../src/model/LLM"

const forwardWithFullRecompute = (llm: LLM, text: string): ReadonlyArray<number> => {
  const tokenized: Array<number> = [...tokenize(text, llm.vocab)]
  const outputTokens: Array<number> = []

  if (tokenized.length === 0 || tokenized.length >= MAX_SEQ_LEN) {
    return outputTokens
  }

  const endTokenId = Option.getOrThrow(llm.vocab.encode("</s>"))

  for (let step = 0; step < MAX_SEQ_LEN - tokenized.length; step++) {
    const tokenInput = T.fromArray(1, tokenized.length, tokenized)
    let input = tokenInput

    for (const layer of llm.network) {
      input = runEffect(layer.forward(input))
    }

    if (input.rows === 0) {
      break
    }

    const lastLogit = runEffect(Ops.rowAsMatrix(input, input.rows - 1))
    const probs = Ops.softmaxRows(lastLogit)
    const tokens = Ops.argmaxRows(probs)
    const nextToken = tokens[tokens.length - 1]!

    outputTokens.push(nextToken)
    tokenized.push(nextToken)

    if (nextToken === endTokenId) {
      break
    }
  }

  return outputTokens
}

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

  test("incremental decoding matches legacy full recompute for standard network", () => {
    const llm = makeLLM({
      vocabWords,
      seed: CANONICAL_SEED,
      numTransformerBlocks: 2
    })

    const actual = runEffect(llm.forward("hello world"))
    const expected = forwardWithFullRecompute(llm, "hello world")

    expect(actual).toEqual(expected)
  })
})
