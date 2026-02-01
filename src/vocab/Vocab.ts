import * as HashMap from "effect/HashMap"
import * as HashSet from "effect/HashSet"
import * as Option from "effect/Option"
import * as Stream from "effect/Stream"
import * as Effect from "effect/Effect"
import { splitWordToTokens } from "../tokenize/split"

export class Vocab {
  readonly encodeMap: HashMap.HashMap<string, number>
  readonly decodeMap: HashMap.HashMap<number, string>
  readonly words: ReadonlyArray<string>

  private constructor(
    encodeMap: HashMap.HashMap<string, number>,
    decodeMap: HashMap.HashMap<number, string>,
    words: ReadonlyArray<string>
  ) {
    this.encodeMap = encodeMap
    this.decodeMap = decodeMap
    this.words = words
  }

  static make(words: ReadonlyArray<string>): Vocab {
    let encodeMap = HashMap.empty<string, number>()
    let decodeMap = HashMap.empty<number, string>()

    for (let i = 0; i < words.length; i += 1) {
      const word = words[i]
      encodeMap = HashMap.set(encodeMap, word, i)
      decodeMap = HashMap.set(decodeMap, i, word)
    }

    return new Vocab(encodeMap, decodeMap, words)
  }

  static defaultWords(): ReadonlyArray<string> {
    return ["hello", "world", "this", "is", "rust", "</s>"]
  }

  encode(word: string): Option.Option<number> {
    return HashMap.get(this.encodeMap, word)
  }

  decode(id: number): Option.Option<string> {
    return HashMap.get(this.decodeMap, id)
  }

  private static addTokensToSet(
    set: HashSet.HashSet<string>,
    text: string
  ): HashSet.HashSet<string> {
    const words = text.split(/\s+/).filter((w) => w.length > 0)
    let next = set

    for (const word of words) {
      const parts = splitWordToTokens(word)
      for (const part of parts) {
        next = HashSet.add(next, part)
      }
    }

    return next
  }

  static processTextForVocab(texts: ReadonlyArray<string>): HashSet.HashSet<string> {
    let vocabSet = HashSet.add(HashSet.empty<string>(), "</s>")
    for (const text of texts) {
      vocabSet = Vocab.addTokensToSet(vocabSet, text)
    }
    return vocabSet
  }

  static processStreamForVocab<E, R>(
    stream: Stream.Stream<string, E, R>
  ): Effect.Effect<HashSet.HashSet<string>, E, R> {
    const initial = HashSet.add(HashSet.empty<string>(), "</s>")
    return Stream.runFold(stream, initial, (set, text) => Vocab.addTokensToSet(set, text))
  }
}
