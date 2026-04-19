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

interface IncrementalEmbeddings {
  forwardTokens(tokenIds: ReadonlyArray<number>): Effect.Effect<Tensor2D, ShapeError>
  forwardToken(tokenId: number, position: number): Effect.Effect<Tensor2D, ShapeError>
}

interface IncrementalTransformerBlock {
  createDecodeState(capacity: number): unknown
  prefill(input: Tensor2D, state: unknown): Effect.Effect<Tensor2D, ShapeError>
  decodeStep(input: Tensor2D, state: unknown): Effect.Effect<Tensor2D, ShapeError>
}

interface IncrementalOutputProjection {
  forwardInference(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError>
}

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

  private static hasIncrementalEmbeddings(layer: ModelLayer): layer is ModelLayer & IncrementalEmbeddings {
    return (
      "forwardTokens" in layer &&
      typeof (layer as { forwardTokens?: unknown }).forwardTokens === "function" &&
      "forwardToken" in layer &&
      typeof (layer as { forwardToken?: unknown }).forwardToken === "function"
    )
  }

  private static hasIncrementalTransformerBlock(
    layer: ModelLayer
  ): layer is ModelLayer & IncrementalTransformerBlock {
    return (
      "createDecodeState" in layer &&
      typeof (layer as { createDecodeState?: unknown }).createDecodeState === "function" &&
      "prefill" in layer &&
      typeof (layer as { prefill?: unknown }).prefill === "function" &&
      "decodeStep" in layer &&
      typeof (layer as { decodeStep?: unknown }).decodeStep === "function"
    )
  }

  private static hasIncrementalOutputProjection(
    layer: ModelLayer
  ): layer is ModelLayer & IncrementalOutputProjection {
    return (
      "forwardInference" in layer &&
      typeof (layer as { forwardInference?: unknown }).forwardInference === "function"
    )
  }

  private isIncrementalNetwork(): boolean {
    if (this.network.length < 2) {
      return false
    }
    if (!LLM.hasIncrementalEmbeddings(this.network[0]!)) {
      return false
    }
    if (!LLM.hasIncrementalOutputProjection(this.network[this.network.length - 1]!)) {
      return false
    }
    for (let i = 1; i < this.network.length - 1; i++) {
      if (!LLM.hasIncrementalTransformerBlock(this.network[i]!)) {
        return false
      }
    }
    return true
  }

  private decodeNextToken(logits: Tensor2D): number {
    const row = logits.rows - 1
    const rowOffset = row * logits.cols
    let maxIdx = 0
    let maxVal = logits.data[rowOffset]!

    for (let col = 1; col < logits.cols; col++) {
      const value = logits.data[rowOffset + col]!
      if (value > maxVal) {
        maxVal = value
        maxIdx = col
      }
    }

    return maxIdx
  }

  private forwardFullRecompute(text: string): Effect.Effect<ReadonlyArray<number>, ShapeError> {
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
        const tokenInput = T.fromArray(1, tokenized.length, tokenized)
        let input: Tensor2D = tokenInput

        for (const layer of this.network) {
          input = yield* layer.forward(input)
        }

        const logits = input
        if (logits.rows === 0) {
          break
        }

        const nextToken = this.decodeNextToken(logits)

        outputTokens.push(nextToken)
        tokenized.push(nextToken)

        if (nextToken === endTokenId.value) {
          break
        }
      }

      return outputTokens
    })
  }

  private forwardIncremental(text: string): Effect.Effect<ReadonlyArray<number>, ShapeError> {
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

      const embeddings = this.network[0]! as ModelLayer & IncrementalEmbeddings
      const outputProjection = this.network[this.network.length - 1]! as ModelLayer &
        IncrementalOutputProjection
      const blocks = this.network.slice(1, -1) as unknown as ReadonlyArray<
        ModelLayer & IncrementalTransformerBlock
      >
      const decodeStates = blocks.map((block) => block.createDecodeState(MAX_SEQ_LEN))

      let hidden: Tensor2D = yield* embeddings.forwardTokens(tokenized)
      for (let i = 0; i < blocks.length; i++) {
        hidden = yield* blocks[i]!.prefill(hidden, decodeStates[i]!)
      }

      const lastHidden = yield* Ops.rowAsMatrix(hidden, hidden.rows - 1)
      let logits: Tensor2D = yield* outputProjection.forwardInference(lastHidden)

      for (let step = 0; step < MAX_SEQ_LEN - inputLen; step++) {
        if (outputTokens.length >= MAX_SEQ_LEN - 1) {
          break
        }

        const nextToken = this.decodeNextToken(logits)
        outputTokens.push(nextToken)
        tokenized.push(nextToken)

        if (nextToken === endTokenId.value) {
          break
        }

        if (tokenized.length >= MAX_SEQ_LEN) {
          break
        }

        hidden = yield* embeddings.forwardToken(nextToken, tokenized.length - 1)
        for (let i = 0; i < blocks.length; i++) {
          hidden = yield* blocks[i]!.decodeStep(hidden, decodeStates[i]!)
        }
        logits = yield* outputProjection.forwardInference(hidden)
      }

      return outputTokens
    })
  }

  forward(text: string): Effect.Effect<ReadonlyArray<number>, ShapeError> {
    return this.isIncrementalNetwork() ? this.forwardIncremental(text) : this.forwardFullRecompute(text)
  }
}
