import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { MAX_SEQ_LEN, EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"

export class Embeddings implements ModelLayer {
  readonly _tag = "Embeddings"
  tokenEmbeddings: Tensor2D
  positionalEmbeddings: Tensor2D

  private cache = new Map<number | string, { tokenIds: Int32Array; positionIds: Int32Array }>()
  private lastCache: { tokenIds: Int32Array; positionIds: Int32Array } | null = null
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

  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const tokenIds = new Int32Array(input.data.length)
      for (let i = 0; i < input.data.length; i++) {
        // Match Rust's float-to-usize truncation behavior.
        tokenIds[i] = Math.trunc(input.data[i])
      }

      const seqLen = tokenIds.length
      const layout = context?.sequenceLayout
      if (layout && layout.totalTokens !== seqLen) {
        return yield* Effect.fail(
          new Ops.ShapeError(`Embeddings.forward: layout totalTokens (${layout.totalTokens}) !== input length (${seqLen})`)
        )
      }

      if (seqLen > this.positionalEmbeddings.rows && !layout) {
        return yield* Effect.fail(
          new Ops.ShapeError(`Sequence length ${seqLen} exceeds maximum ${this.positionalEmbeddings.rows}`)
        )
      }

      const positionIds = layout
        ? new Int32Array(layout.positionIds)
        : Int32Array.from({ length: seqLen }, (_, i) => i)
      this.cache.set(key, { tokenIds, positionIds })
      this.lastCache = { tokenIds, positionIds }

      const tokenEmbeds = yield* Ops.gatherRows(this.tokenEmbeddings, tokenIds)
      const posEmbeds = layout
        ? yield* Ops.gatherRows(this.positionalEmbeddings, positionIds)
        : yield* Ops.sliceRows(this.positionalEmbeddings, 0, seqLen)
      const combined = yield* Ops.add(tokenEmbeds, posEmbeds)
      return combined
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = this.cache.get(key) ?? this.lastCache
      if (!cached) {
        return yield* Effect.fail(new Ops.ShapeError("Embeddings.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const { tokenIds, positionIds } = cached
      const cols = dOut.cols

      const tokenRowToGradIndex = new Map<number, number>()
      const tokenRows: Array<number> = []
      const positionRowToGradIndex = new Map<number, number>()
      const positionRows: Array<number> = []

      for (let i = 0; i < tokenIds.length; i++) {
        const tokenId = tokenIds[i]
        if (tokenId < 0 || tokenId >= this.tokenEmbeddings.rows) {
          return yield* Effect.fail(
            new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${this.tokenEmbeddings.rows}`)
          )
        }
        const positionId = positionIds[i]!
        if (positionId < 0 || positionId >= this.positionalEmbeddings.rows) {
          return yield* Effect.fail(
            new Ops.ShapeError(
              `Position ID ${positionId} out of bounds for max sequence length ${this.positionalEmbeddings.rows}`
            )
          )
        }
        if (!tokenRowToGradIndex.has(tokenId)) {
          tokenRowToGradIndex.set(tokenId, tokenRows.length)
          tokenRows.push(tokenId)
        }
        if (!positionRowToGradIndex.has(positionId)) {
          positionRowToGradIndex.set(positionId, positionRows.length)
          positionRows.push(positionId)
        }
      }

      const tokenGradRows = new Float32Array(tokenRows.length * cols)
      const positionGradRows = new Float32Array(positionRows.length * cols)

      for (let i = 0; i < tokenIds.length; i++) {
        const tokenGradRow = tokenRowToGradIndex.get(tokenIds[i])!
        const positionGradRow = positionRowToGradIndex.get(positionIds[i]!)!
        const rowOffset = i * cols
        const tokenOffset = tokenGradRow * cols
        const positionOffset = positionGradRow * cols
        for (let j = 0; j < cols; j++) {
          const grad = dOut.data[rowOffset + j]
          tokenGradRows[tokenOffset + j] += grad
          positionGradRows[positionOffset + j] += grad
        }
      }

      this.tokenOptimizer.stepRows(this.tokenEmbeddings, tokenRows, tokenGradRows, lr)
      this.positionalOptimizer.stepRows(this.positionalEmbeddings, positionRows, positionGradRows, lr)

      return dOut
    })
  }
}
