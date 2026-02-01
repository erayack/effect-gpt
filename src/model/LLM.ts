import * as Effect from "effect/Effect"
import * as Option from "effect/Option"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { Embeddings } from "./Embeddings"
import { TransformerBlock } from "./TransformerBlock"
import { OutputProjection } from "./OutputProjection"
import { Vocab } from "../vocab/Vocab"
import { tokenize } from "../tokenize/tokenize"
import { MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import type { Rng } from "../tensor/random"

export class LLM {
  readonly vocab: Vocab
  readonly network: ReadonlyArray<ModelLayer>

  constructor(vocab: Vocab, network: ReadonlyArray<ModelLayer>) {
    this.vocab = vocab
    this.network = network
  }

  static default(rng: Rng, numTransformerBlocks = 1): LLM {
    const vocab = Vocab.make(Vocab.defaultWords())
    return LLM.make(vocab, rng, numTransformerBlocks)
  }

  static make(vocab: Vocab, rng: Rng, numTransformerBlocks = 3): LLM {
    const vocabSize = vocab.words.length
    const network: Array<ModelLayer> = [
      new Embeddings(vocabSize, EMBEDDING_DIM, MAX_SEQ_LEN, rng),
      ...Array.from({ length: numTransformerBlocks }, () => new TransformerBlock(EMBEDDING_DIM, HIDDEN_DIM, rng)),
      new OutputProjection(EMBEDDING_DIM, vocabSize, rng)
    ]
    return new LLM(vocab, network)
  }

  networkDescription(): string {
    return this.network.map((layer) => layer._tag).join(", ")
  }

  totalParameters(): number {
    return this.network.reduce((sum, layer) => sum + layer.parametersCount, 0)
  }

  predict(text: string): Effect.Effect<string, ShapeError> {
    return Effect.gen(this, function* () {
      const outputTokens = yield* this.forward(text)

      if (outputTokens.length === 0) {
        return ""
      }

      const tokenStrs: Array<string> = []
      for (const t of outputTokens) {
        const decoded = this.vocab.decode(t)
        if (Option.isSome(decoded)) {
          tokenStrs.push(decoded.value)
        }
      }

      return tokenStrs.join(" ")
    })
  }

  forward(text: string): Effect.Effect<ReadonlyArray<number>, ShapeError> {
    return Effect.gen(this, function* () {
      const tokenized: Array<number> = [...tokenize(text, this.vocab)]
      const outputTokens: Array<number> = []

      if (tokenized.length === 0) {
        return outputTokens
      }

      const inputLen = tokenized.length
      if (inputLen >= MAX_SEQ_LEN) {
        return outputTokens
      }

      const endTokenId = this.vocab.encode("</s>")
      if (Option.isNone(endTokenId)) {
        return yield* Effect.fail(new Ops.ShapeError("End token </s> not found in vocabulary"))
      }

      for (let step = 0; step < MAX_SEQ_LEN - inputLen; step++) {
        if (outputTokens.length >= MAX_SEQ_LEN - 1) {
          break
        }

        const tokenInput = T.fromArray(1, tokenized.length, tokenized)
        let input: Tensor2D = tokenInput

        for (const layer of this.network) {
          input = yield* layer.forward(input)
        }

        const logits = input

        if (logits.rows === 0) {
          break
        }

        const lastRowStart = (logits.rows - 1) * logits.cols
        const lastLogitData = logits.data.slice(lastRowStart, lastRowStart + logits.cols)
        const lastLogit = T.make(1, logits.cols, lastLogitData)

        const probs = Ops.softmaxRows(lastLogit)
        const tokens = Ops.argmaxRows(probs)
        const nextToken = tokens[tokens.length - 1]

        outputTokens.push(nextToken)
        tokenized.push(nextToken)

        if (nextToken === endTokenId.value) {
          break
        }
      }

      return outputTokens
    })
  }
}
