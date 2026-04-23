import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { MAX_SEQ_LEN, EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

export interface EmbeddingsInferenceState {
  readonly workspace: TensorWorkspace
}

interface EmbeddingsCacheEntry {
  readonly workspace: TensorWorkspace
  readonly tokenIds: Int32Array
  readonly positionIds: Int32Array
}

export class Embeddings implements SyncModelLayer {
  readonly _tag = "Embeddings"
  tokenEmbeddings: Tensor2D
  positionalEmbeddings: Tensor2D

  private cache = new Map<LayerCacheKey, EmbeddingsCacheEntry>()
  private lastCache: EmbeddingsCacheEntry | null = null
  private readonly workspacePool: Array<TensorWorkspace> = []
  tokenOptimizer: Adam
  positionalOptimizer: Adam

  constructor(vocabSize: number, embeddingDim: number = EMBEDDING_DIM, maxSeqLen: number = MAX_SEQ_LEN, rng: Rng) {
    this.tokenEmbeddings = Ops.initNormal(vocabSize, embeddingDim, 0, 0.02, rng)
    this.positionalEmbeddings = Ops.initNormal(maxSeqLen, embeddingDim, 0, 0.02, rng)
    this.tokenOptimizer = Adam.make(vocabSize, embeddingDim)
    this.positionalOptimizer = Adam.make(maxSeqLen, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): LayerCacheKey {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.tokenEmbeddings.data.length + this.positionalEmbeddings.data.length
  }

  private acquireWorkspace(): TensorWorkspace {
    return this.workspacePool.pop() ?? new TensorWorkspace()
  }

  private releaseWorkspace(workspace: TensorWorkspace): void {
    this.workspacePool.push(workspace)
  }

  private storeCache(
    cacheKey: LayerCacheKey | undefined,
    workspace: TensorWorkspace,
    tokenIds: Int32Array,
    positionIds: Int32Array
  ): void {
    if (cacheKey !== undefined && this.cache.has(cacheKey)) {
      throw new Ops.ShapeError("Embeddings.forward received a cacheKey that is already active")
    }
    const entry = { workspace, tokenIds, positionIds }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, entry)
    }
    this.lastCache = entry
  }

  // Inference-only embedding lookup that does not touch the training cache.
  // Used by KV-cache prefill/decode in LLM.forwardIncremental.
  private embedTokenIdsSync(
    tokenIds: ReadonlyArray<number>,
    startPosition: number,
    state?: EmbeddingsInferenceState
  ): Tensor2D {
    const seqLen = tokenIds.length
    const endPosition = startPosition + seqLen
    if (endPosition > this.positionalEmbeddings.rows) {
      throw new Ops.ShapeError(`Sequence length ${endPosition} exceeds maximum ${this.positionalEmbeddings.rows}`)
    }
    const vocabRows = this.tokenEmbeddings.rows
    for (let i = 0; i < seqLen; i++) {
      const tokenId = tokenIds[i]!
      if (tokenId < 0 || tokenId >= vocabRows) {
        throw new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${vocabRows}`)
      }
    }

    const workspace = state?.workspace ?? new TensorWorkspace()
    const combined = workspace.borrowTensor("combined", seqLen, this.tokenEmbeddings.cols)

    this.embedContiguousPositionsInto(tokenIds, startPosition, combined)
    return combined
  }

  private embedContiguousPositionsInto(tokenIds: ArrayLike<number>, startPosition: number, out: Tensor2D): void {
    const rows = tokenIds.length
    const cols = this.tokenEmbeddings.cols
    const tokenData = this.tokenEmbeddings.data
    const positionData = this.positionalEmbeddings.data
    const outData = out.data
    let positionOffset = startPosition * cols
    let outOffset = 0

    for (let row = 0; row < rows; row++) {
      const tokenOffset = tokenIds[row] * cols
      for (let col = 0; col < cols; col++) {
        outData[outOffset + col] = tokenData[tokenOffset + col] + positionData[positionOffset + col]
      }
      positionOffset += cols
      outOffset += cols
    }
  }

  private embedPositionIdsInto(tokenIds: ArrayLike<number>, positionIds: Int32Array, out: Tensor2D): void {
    const rows = tokenIds.length
    const cols = this.tokenEmbeddings.cols
    const tokenData = this.tokenEmbeddings.data
    const positionData = this.positionalEmbeddings.data
    const outData = out.data
    let outOffset = 0

    for (let row = 0; row < rows; row++) {
      const tokenOffset = tokenIds[row] * cols
      const positionOffset = positionIds[row] * cols
      for (let col = 0; col < cols; col++) {
        outData[outOffset + col] = tokenData[tokenOffset + col] + positionData[positionOffset + col]
      }
      outOffset += cols
    }
  }

  forwardTokensSync(tokenIds: ReadonlyArray<number>, state?: EmbeddingsInferenceState): Tensor2D {
    return this.embedTokenIdsSync(tokenIds, 0, state)
  }

  forwardTokens(tokenIds: ReadonlyArray<number>): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.forwardTokensSync(tokenIds))
  }

  forwardTokenSync(tokenId: number, position: number, state?: EmbeddingsInferenceState): Tensor2D {
    if (tokenId < 0 || tokenId >= this.tokenEmbeddings.rows) {
      throw new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${this.tokenEmbeddings.rows}`)
    }
    if (position < 0 || position >= this.positionalEmbeddings.rows) {
      throw new Ops.ShapeError(`Position ID ${position} out of bounds for max sequence length ${this.positionalEmbeddings.rows}`)
    }

    const workspace = state?.workspace ?? new TensorWorkspace()
    const combined = workspace.borrowTensor("combined", 1, this.tokenEmbeddings.cols)
    const cols = this.tokenEmbeddings.cols
    const tokenOffset = tokenId * cols
    const positionOffset = position * cols
    const tokenData = this.tokenEmbeddings.data
    const positionData = this.positionalEmbeddings.data
    const outData = combined.data

    for (let col = 0; col < cols; col++) {
      outData[col] = tokenData[tokenOffset + col] + positionData[positionOffset + col]
    }

    return combined
  }

  forwardToken(tokenId: number, position: number): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.forwardTokenSync(tokenId, position))
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    const tokenCount = input.data.length
    const tokenIds = new Int32Array(tokenCount)
    const vocabRows = this.tokenEmbeddings.rows
    for (let i = 0; i < tokenCount; i++) {
      // Match Rust's float-to-usize truncation behavior.
      const tokenId = Math.trunc(input.data[i])
      if (tokenId < 0 || tokenId >= vocabRows) {
        throw new Ops.ShapeError(`Token ID ${tokenId} out of bounds for vocab size ${vocabRows}`)
      }
      tokenIds[i] = tokenId
    }

    const seqLen = tokenIds.length
    const layout = context?.sequenceLayout
    if (layout && layout.totalTokens !== seqLen) {
      throw new Ops.ShapeError(`Embeddings.forward: layout totalTokens (${layout.totalTokens}) !== input length (${seqLen})`)
    }

    if (seqLen > this.positionalEmbeddings.rows && !layout) {
      throw new Ops.ShapeError(`Sequence length ${seqLen} exceeds maximum ${this.positionalEmbeddings.rows}`)
    }

    const positionIds = new Int32Array(seqLen)
    if (layout) {
      if (layout.positionIds.length !== seqLen) {
        throw new Ops.ShapeError(
          `Embeddings.forward: layout positionIds length (${layout.positionIds.length}) !== input length (${seqLen})`
        )
      }
      const maxPositionRows = this.positionalEmbeddings.rows
      for (let i = 0; i < seqLen; i++) {
        const positionId = layout.positionIds[i]!
        if (positionId < 0 || positionId >= maxPositionRows) {
          throw new Ops.ShapeError(
            `Position ID ${positionId} out of bounds for max sequence length ${maxPositionRows}`
          )
        }
        positionIds[i] = positionId
      }
    } else {
      for (let i = 0; i < seqLen; i++) {
        positionIds[i] = i
      }
    }

    const captureCache = context?.captureCache !== false
    const workspace = captureCache ? this.acquireWorkspace() : new TensorWorkspace()
    let stored = false
    try {
      const combined = workspace.borrowTensor("combined", seqLen, this.tokenEmbeddings.cols)

      if (layout) {
        this.embedPositionIdsInto(tokenIds, positionIds, combined)
      } else {
        this.embedContiguousPositionsInto(tokenIds, 0, combined)
      }
      if (captureCache) {
        this.storeCache(context?.cacheKey, workspace, tokenIds, positionIds)
        stored = true
      }
      return combined
    } catch (error) {
      if (captureCache && !stored) {
        this.releaseWorkspace(workspace)
      }
      throw error
    }
  }

  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() =>
        this.forwardSync(input, {
          ...context,
          cacheKey: context?.cacheKey ?? this.fiberKey(fiberId),
          captureCache: context?.captureCache ?? true
        })
      )
    })
  }

  backwardSync(dOut: Tensor2D, lr: number, cacheKey?: LayerCacheKey): Tensor2D {
    const cached = (cacheKey !== undefined ? this.cache.get(cacheKey) : undefined) ?? this.lastCache
    if (!cached) {
      throw new Ops.ShapeError("Embeddings.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { workspace, tokenIds, positionIds } = cached
    try {
      const cols = dOut.cols

      const tokenRowToGradIndex = new Map<number, number>()
      const tokenRows: Array<number> = []
      const positionRowToGradIndex = new Map<number, number>()
      const positionRows: Array<number> = []

      const tokenCount = tokenIds.length
      for (let i = 0; i < tokenCount; i++) {
        const tokenId = tokenIds[i]
        const positionId = positionIds[i]!
        let tokenGradIndex = tokenRowToGradIndex.get(tokenId)
        if (tokenGradIndex === undefined) {
          tokenGradIndex = tokenRows.length
          tokenRowToGradIndex.set(tokenId, tokenGradIndex)
          tokenRows.push(tokenId)
        }
        let positionGradIndex = positionRowToGradIndex.get(positionId)
        if (positionGradIndex === undefined) {
          positionGradIndex = positionRows.length
          positionRowToGradIndex.set(positionId, positionGradIndex)
          positionRows.push(positionId)
        }
      }

      const tokenGradRows = new Float32Array(tokenRows.length * cols)
      const positionGradRows = new Float32Array(positionRows.length * cols)
      const dOutData = dOut.data

      for (let i = 0; i < tokenCount; i++) {
        const tokenGradRow = tokenRowToGradIndex.get(tokenIds[i])!
        const positionGradRow = positionRowToGradIndex.get(positionIds[i]!)!
        const rowOffset = i * cols
        const tokenOffset = tokenGradRow * cols
        const positionOffset = positionGradRow * cols
        for (let j = 0; j < cols; j++) {
          const grad = dOutData[rowOffset + j]
          tokenGradRows[tokenOffset + j] += grad
          positionGradRows[positionOffset + j] += grad
        }
      }

      this.tokenOptimizer.stepRows(this.tokenEmbeddings, tokenRows, tokenGradRows, lr)
      this.positionalOptimizer.stepRows(this.positionalEmbeddings, positionRows, positionGradRows, lr)

      return dOut
    } finally {
      this.releaseWorkspace(workspace)
    }
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
