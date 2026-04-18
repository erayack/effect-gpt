import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
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

  private cache = new Map<number | string, Tensor2D>()
  private lastCache: Tensor2D | null = null
  tokenOptimizer: Adam
  positionalOptimizer: Adam

  constructor(vocabSize: number, embeddingDim: number = EMBEDDING_DIM, maxSeqLen: number = MAX_SEQ_LEN, rng: Rng) {
    this.tokenEmbeddings = Ops.initNormal(vocabSize, embeddingDim, 0, 0.02, rng)
    this.positionalEmbeddings = Ops.initNormal(maxSeqLen, embeddingDim, 0, 0.02, rng)
    this.tokenOptimizer = Adam.make(vocabSize, embeddingDim)
    this.positionalOptimizer = Adam.make(maxSeqLen, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.tokenEmbeddings.data.length + this.positionalEmbeddings.data.length
  }

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cloned = T.clone(input)
      this.cache.set(key, cloned)
      this.lastCache = cloned
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
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cachedInput = this.cache.get(key) ?? this.lastCache
      if (!cachedInput) {
        return yield* Effect.fail(new Ops.ShapeError("Embeddings.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const input = cachedInput
      const tokenIds: Array<number> = []
      for (let i = 0; i < input.data.length; i++) {
        tokenIds.push(Math.trunc(input.data[i]))
      }

      const cols = dOut.cols
      const tokenRowToGradIndex = new Map<number, number>()
      const tokenRows: Array<number> = []
      let uniqueTokenRows = 0

      for (let i = 0; i < tokenIds.length; i++) {
        const tokenId = tokenIds[i]
        if (tokenId < 0 || tokenId >= this.tokenEmbeddings.rows) {
          return yield* Effect.fail(
            new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${this.tokenEmbeddings.rows}`)
          )
        }
        if (!tokenRowToGradIndex.has(tokenId)) {
          tokenRowToGradIndex.set(tokenId, uniqueTokenRows++)
          tokenRows.push(tokenId)
        }
      }

      const tokenGradRows = new Float32Array(uniqueTokenRows * cols)
      for (let i = 0; i < tokenIds.length; i++) {
        const gradRow = tokenRowToGradIndex.get(tokenIds[i])!
        const rowOffset = i * cols
        const tokenOffset = gradRow * cols
        for (let j = 0; j < cols; j++) {
          tokenGradRows[tokenOffset + j] += dOut.data[rowOffset + j]
        }
      }

      const positionalRows = Array.from({ length: tokenIds.length }, (_, i) => i)
      this.tokenOptimizer.stepRows(this.tokenEmbeddings, tokenRows, tokenGradRows, lr)
      this.positionalOptimizer.stepRows(this.positionalEmbeddings, positionalRows, dOut.data, lr)

      return T.clone(dOut)
    })
  }
}
