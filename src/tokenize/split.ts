const ASCII_PUNCTUATION = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/

export const isAsciiPunctuation = (ch: string): boolean =>
  ch.length === 1 && ASCII_PUNCTUATION.test(ch)

export const splitWordToTokens = (word: string): ReadonlyArray<string> => {
  const tokens: Array<string> = []
  let current = ""

  for (const ch of word) {
    if (isAsciiPunctuation(ch)) {
      if (current.length > 0) {
        tokens.push(current)
        current = ""
      }
      tokens.push(ch)
    } else {
      current += ch
    }
  }

  if (current.length > 0) {
    tokens.push(current)
  }

  return tokens
}
