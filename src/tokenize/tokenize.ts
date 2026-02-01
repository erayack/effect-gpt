import * as Option from "effect/Option"
import { splitWordToTokens } from "./split"
import { Vocab } from "../vocab/Vocab"

export const tokenize = (text: string, vocab: Vocab): ReadonlyArray<number> => {
  const tokens: Array<number> = []
  const words = text.split(/\s+/).filter((w) => w.length > 0)

  for (const word of words) {
    if (word === "</s>") {
      const tokenId = vocab.encode(word)
      if (Option.isSome(tokenId)) {
        tokens.push(tokenId.value as number)
      }
      continue
    }

    const split = splitWordToTokens(word)
    for (const part of split) {
      const tokenId = vocab.encode(part)
      if (Option.isSome(tokenId)) {
        tokens.push(tokenId.value as number)
      }
    }
  }

  return tokens
}
