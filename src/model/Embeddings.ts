import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { MAX_SEQ_LEN, EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

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

  // Inference-only embedding lookup that does not touch the training cache.
  // Used by KV-cache prefill/decode in LLM.forwardIncremental.
  private embedTokenIds(
    tokenIds: ReadonlyArray<number>,
    startPosition: number
  ): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const seqLen = tokenIds.length
      const endPosition = startPosition + seqLen
      if (endPosition > this.positionalEmbeddings.rows) {
        return yield* Effect.fail(
          new Ops.ShapeError(`Sequence length ${endPosition} exceeds maximum ${this.positionalEmbeddings.rows}`)
        )
      }

      const workspace = new TensorWorkspace()
      const tokenEmbeds = workspace.borrowTensor("tokenEmbeds", tokenIds.length, this.tokenEmbeddings.cols)
      const posEmbeds = workspace.borrowTensor("posEmbeds", tokenIds.length, this.positionalEmbeddings.cols)
      const combined = T.zeros(tokenIds.length, this.tokenEmbeddings.cols)

      yield* Ops.gatherRowsInto(this.tokenEmbeddings, tokenIds, tokenEmbeds)
      yield* Ops.sliceRowsInto(this.positionalEmbeddings, startPosition, endPosition, posEmbeds)
      yield* Ops.addInto(tokenEmbeds, posEmbeds, combined)
      return combined
    })
  }

  forwardTokens(tokenIds: ReadonlyArray<number>): Effect.Effect<Tensor2D, ShapeError> {
    return this.embedTokenIds(tokenIds, 0)
  }

  forwardToken(tokenId: number, position: number): Effect.Effect<Tensor2D, ShapeError> {
    return this.embedTokenIds([tokenId], position)
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

      const workspace = new TensorWorkspace()
      const tokenEmbeds = workspace.borrowTensor("tokenEmbeds", seqLen, this.tokenEmbeddings.cols)
      const posEmbeds = workspace.borrowTensor("posEmbeds", seqLen, this.positionalEmbeddings.cols)
      const combined = T.zeros(seqLen, this.tokenEmbeddings.cols)

      yield* Ops.gatherRowsInto(this.tokenEmbeddings, tokenIds, tokenEmbeds)
      if (layout) {
        yield* Ops.gatherRowsInto(this.positionalEmbeddings, positionIds, posEmbeds)
      } else {
        yield* Ops.sliceRowsInto(this.positionalEmbeddings, 0, seqLen, posEmbeds)
      }
      yield* Ops.addInto(tokenEmbeds, posEmbeds, combined)
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
