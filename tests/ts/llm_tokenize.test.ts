import { describe, test, expect } from "bun:test"
import { Vocab } from "../../src/vocab/Vocab"
import { tokenize } from "../../src/tokenize/tokenize"
import * as Option from "effect/Option"

describe("LLM Tokenize", () => {
  const vocab = Vocab.make(Vocab.defaultWords())

  test("encode single word", () => {
    const tokens = tokenize("hello", vocab)
    expect(tokens.length).toBe(1)
    expect(tokens[0]).toBe(Option.getOrThrow(vocab.encode("hello")))
  })

  test("encode multiple words", () => {
    const tokens = tokenize("hello world", vocab)
    expect(tokens.length).toBe(2)
    expect(tokens[0]).toBe(Option.getOrThrow(vocab.encode("hello")))
    expect(tokens[1]).toBe(Option.getOrThrow(vocab.encode("world")))
  })

  test("</s> token encodes correctly", () => {
    const eosId = vocab.encode("</s>")
    expect(Option.isSome(eosId)).toBe(true)
    const tokens = tokenize("</s>", vocab)
    expect(tokens.length).toBe(1)
    expect(tokens[0]).toBe(Option.getOrThrow(eosId))
  })

  test("decode returns original word", () => {
    const tokens = tokenize("hello", vocab)
    const decoded = vocab.decode(tokens[0]!)
    expect(Option.isSome(decoded)).toBe(true)
    expect(Option.getOrThrow(decoded)).toBe("hello")
  })

  test("roundtrip encode/decode for all default words", () => {
    const words = Vocab.defaultWords()
    for (const word of words) {
      const tokens = tokenize(word, vocab)
      expect(tokens.length).toBeGreaterThanOrEqual(1)
      const decoded = vocab.decode(tokens[0]!)
      expect(Option.isSome(decoded)).toBe(true)
    }
  })

  test("sentence with </s> appended encodes correctly", () => {
    const tokens = tokenize("hello world </s>", vocab)
    expect(tokens.length).toBe(3)
    const lastToken = tokens[tokens.length - 1]!
    const eosId = Option.getOrThrow(vocab.encode("</s>"))
    expect(lastToken).toBe(eosId)
  })

  test("empty string returns empty tokens", () => {
    const tokens = tokenize("", vocab)
    expect(tokens.length).toBe(0)
  })

  test("unknown word returns empty for that word", () => {
    const tokens = tokenize("unknownxyz", vocab)
    expect(tokens.length).toBe(0)
  })

  test("multiple spaces handled correctly", () => {
    const tokens = tokenize("hello   world", vocab)
    expect(tokens.length).toBe(2)
  })
})
