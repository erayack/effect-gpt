import * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { MAX_SEQ_LEN, EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"

export class Embeddings implements ModelLayer {
  readonly _tag = "Embeddings"
  tokenEmbeddings: Tensor2D
  positionalEmbeddings: Tensor2D

  cachedInput: Tensor2D | null = null
  tokenOptimizer: Adam
  positionalOptimizer: Adam

  constructor(vocabSize: number, embeddingDim: number = EMBEDDING_DIM, maxSeqLen: number = MAX_SEQ_LEN, rng?: Rng) {
    this.tokenEmbeddings = Ops.initNormal(vocabSize, embeddingDim, 0, 0.02, rng)
    this.positionalEmbeddings = Ops.initNormal(maxSeqLen, embeddingDim, 0, 0.02, rng)
    this.tokenOptimizer = Adam.make(vocabSize, embeddingDim)
    this.positionalOptimizer = Adam.make(maxSeqLen, embeddingDim)
  }

  get parametersCount(): number {
    return this.tokenEmbeddings.data.length + this.positionalEmbeddings.data.length
  }

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      this.cachedInput = T.clone(input)
      const tokenIds: Array<number> = []
      for (let i = 0; i < input.data.length; i++) {
        // Match Rust's float-to-usize truncation behavior.
        tokenIds.push(Math.trunc(input.data[i]))
      }

      const seqLen = tokenIds.length
      if (seqLen > this.positionalEmbeddings.rows) {
        return yield* Effect.fail(
          new Ops.ShapeError(`Sequence length ${seqLen} exceeds maximum ${this.positionalEmbeddings.rows}`)
        )
      }

      const tokenEmbeds = yield* Ops.gatherRows(this.tokenEmbeddings, tokenIds)
      const posEmbeds = yield* Ops.sliceRows(this.positionalEmbeddings, 0, seqLen)
      const combined = yield* Ops.add(tokenEmbeds, posEmbeds)
      return combined
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      if (!this.cachedInput) {
        return yield* Effect.fail(new Ops.ShapeError("Embeddings.backward called before forward"))
      }

      const input = this.cachedInput
      const tokenIds: Array<number> = []
      for (let i = 0; i < input.data.length; i++) {
        tokenIds.push(Math.trunc(input.data[i]))
      }

      const tokenGrads = T.zeros(this.tokenEmbeddings.rows, this.tokenEmbeddings.cols)
      const positionalGrads = T.zeros(this.positionalEmbeddings.rows, this.positionalEmbeddings.cols)

      const seqLen = tokenIds.length
      for (let i = 0; i < seqLen; i++) {
        const tokenId = tokenIds[i]
        if (tokenId < 0 || tokenId >= this.tokenEmbeddings.rows) {
          return yield* Effect.fail(
            new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${this.tokenEmbeddings.rows}`)
          )
        }
        const rowOffset = i * dOut.cols
        const tokenOffset = tokenId * tokenGrads.cols
        const posOffset = i * positionalGrads.cols
        for (let j = 0; j < dOut.cols; j++) {
          const grad = dOut.data[rowOffset + j]
          tokenGrads.data[tokenOffset + j] += grad
          positionalGrads.data[posOffset + j] += grad
        }
      }

      this.tokenOptimizer.step(this.tokenEmbeddings, tokenGrads, lr)
      this.positionalOptimizer.step(this.positionalEmbeddings, positionalGrads, lr)

      return T.clone(dOut)
    })
  }
}
