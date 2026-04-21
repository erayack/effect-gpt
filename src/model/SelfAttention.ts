import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SequenceLayout, SyncModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

export interface SelfAttentionKvCache {
  readonly keys: Float32Array
  readonly values: Float32Array
  readonly capacity: number
  readonly embeddingDim: number
  length: number
}

export class SelfAttention implements SyncModelLayer {
  readonly _tag = "SelfAttention"
  readonly embeddingDim: number
  wQKV: Tensor2D

  private cache = new Map<LayerCacheKey, { input: Tensor2D; q: Tensor2D; k: Tensor2D; v: Tensor2D; attnWeights: Tensor2D }>()
  private lastCache: { input: Tensor2D; q: Tensor2D; k: Tensor2D; v: Tensor2D; attnWeights: Tensor2D } | null = null
  private readonly projectionOptimizer: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, rng: Rng) {
    this.embeddingDim = embeddingDim
    const std = Math.sqrt(2.0 / embeddingDim)
    this.wQKV = Ops.initNormal(embeddingDim, embeddingDim * 3, 0, std, rng)
    this.projectionOptimizer = Adam.make(embeddingDim, embeddingDim * 3)
  }

  private fiberKey(fiberId: FiberId.FiberId): LayerCacheKey {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.wQKV.data.length
  }

  private storeCache(
    cacheKey: LayerCacheKey | undefined,
    input: Tensor2D,
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    attnWeights: Tensor2D
  ): void {
    const cached = { input, q, k, v, attnWeights }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, cached)
    }
    this.lastCache = cached
  }

  private unpackProjectedQKV(projectedQKV: Tensor2D, q: Tensor2D, k: Tensor2D, v: Tensor2D): void {
    const dim = this.embeddingDim
    const projectedData = projectedQKV.data
    const qData = q.data
    const kData = k.data
    const vData = v.data

    for (let row = 0; row < projectedQKV.rows; row++) {
      const projectedOffset = row * dim * 3
      const rowOffset = row * dim
      for (let col = 0; col < dim; col++) {
        qData[rowOffset + col] = projectedData[projectedOffset + col]!
        kData[rowOffset + col] = projectedData[projectedOffset + dim + col]!
        vData[rowOffset + col] = projectedData[projectedOffset + dim * 2 + col]!
      }
    }
  }

  private packGradientsSync(gradQ: Tensor2D, gradK: Tensor2D, gradV: Tensor2D, out: Tensor2D): void {
    const dim = this.embeddingDim
    const outData = out.data
    const qData = gradQ.data
    const kData = gradK.data
    const vData = gradV.data

    for (let row = 0; row < out.rows; row++) {
      const projectedOffset = row * dim * 3
      const rowOffset = row * dim
      for (let col = 0; col < dim; col++) {
        outData[projectedOffset + col] = qData[rowOffset + col]!
        outData[projectedOffset + dim + col] = kData[rowOffset + col]!
        outData[projectedOffset + dim * 2 + col] = vData[rowOffset + col]!
      }
    }
  }

  private computeQKVSync(
    input: Tensor2D,
    workspace: TensorWorkspace
  ): { q: Tensor2D; k: Tensor2D; v: Tensor2D } {
    const projectedQKV = workspace.borrowTensor("projectedQKV", input.rows, this.embeddingDim * 3)
    const q = workspace.borrowTensor("q", input.rows, this.embeddingDim)
    const k = workspace.borrowTensor("k", input.rows, this.embeddingDim)
    const v = workspace.borrowTensor("v", input.rows, this.embeddingDim)
    Ops.matMulIntoSync(input, this.wQKV, projectedQKV, { workspace })
    this.unpackProjectedQKV(projectedQKV, q, k, v)
    return { q, k, v }
  }

  private attentionSync(
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    workspace: TensorWorkspace,
    layout?: SequenceLayout
  ): { attnWeights: Tensor2D; attended: Tensor2D } {
    const dk = Math.sqrt(this.embeddingDim)
    const scores = workspace.borrowTensor("scores", q.rows, k.rows)
    Ops.matMulIntoSync(q, k, scores, { transposeB: true, workspace })
    Ops.mulScalarInPlace(scores, 1 / dk)
    Ops.maskCausalInPlace(scores, layout)

    const attnWeights = workspace.borrowTensor("attnWeights", scores.rows, scores.cols)
    Ops.softmaxRowsInto(scores, attnWeights)
    const attended = workspace.borrowTensor("attended", attnWeights.rows, v.cols)
    Ops.matMulIntoSync(attnWeights, v, attended, { workspace })
    return { attnWeights, attended }
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    const workspace = new TensorWorkspace()
    const { q, k, v } = this.computeQKVSync(input, workspace)
    const { attnWeights, attended } = this.attentionSync(q, k, v, workspace, context?.sequenceLayout)
    if (context?.captureCache !== false) {
      this.storeCache(context?.cacheKey, input, q, k, v, attnWeights)
    }
    return Ops.addSync(attended, input)
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

  createKvCache(capacity: number): SelfAttentionKvCache {
    return {
      keys: new Float32Array(capacity * this.embeddingDim),
      values: new Float32Array(capacity * this.embeddingDim),
      capacity,
      embeddingDim: this.embeddingDim,
      length: 0
    }
  }

  private validateKvCacheSync(cache: SelfAttentionKvCache): void {
    if (cache.embeddingDim !== this.embeddingDim) {
      throw new Ops.ShapeError(
        `KV cache embedding dimension ${cache.embeddingDim} does not match attention dimension ${this.embeddingDim}`
      )
    }
    if (cache.length < 0 || cache.length > cache.capacity) {
      throw new Ops.ShapeError(`KV cache length ${cache.length} is out of bounds for capacity ${cache.capacity}`)
    }
  }

  private ensureEmptyKvCacheSync(cache: SelfAttentionKvCache): void {
    if (cache.length !== 0) {
      throw new Ops.ShapeError(
        `KV cache must be empty before prefill, received length ${cache.length}. Create a fresh cache for each prompt.`
      )
    }
  }

  private storeKvRowsSync(k: Tensor2D, v: Tensor2D, cache: SelfAttentionKvCache): void {
    const nextLength = cache.length + k.rows
    if (nextLength > cache.capacity) {
      throw new Ops.ShapeError(`KV cache capacity ${cache.capacity} exceeded by sequence length ${nextLength}`)
    }

    const offset = cache.length * this.embeddingDim
    cache.keys.set(k.data, offset)
    cache.values.set(v.data, offset)
    cache.length = nextLength
  }

  prefillSync(input: Tensor2D, cache: SelfAttentionKvCache): Tensor2D {
    this.validateKvCacheSync(cache)
    this.ensureEmptyKvCacheSync(cache)
    const workspace = new TensorWorkspace()
    const { q, k, v } = this.computeQKVSync(input, workspace)
    this.storeKvRowsSync(k, v, cache)
    const { attended } = this.attentionSync(q, k, v, workspace)
    return Ops.addSync(attended, input)
  }

  prefill(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.prefillSync(input, cache))
  }

  decodeStepSync(input: Tensor2D, cache: SelfAttentionKvCache): Tensor2D {
    this.validateKvCacheSync(cache)
    if (input.rows !== 1) {
      throw new Ops.ShapeError(`decodeStep expects a single row, received ${input.rows}`)
    }

    const workspace = new TensorWorkspace()
    const { q, k, v } = this.computeQKVSync(input, workspace)
    this.storeKvRowsSync(k, v, cache)

    const seqLen = cache.length
    const dim = this.embeddingDim
    const scale = Math.sqrt(dim)
    const scores = new Float32Array(seqLen)
    let maxScore = -Infinity
    for (let row = 0; row < seqLen; row++) {
      const cacheOffset = row * dim
      let score = 0
      for (let col = 0; col < dim; col++) {
        score += q.data[col] * cache.keys[cacheOffset + col]
      }
      score /= scale
      scores[row] = score
      if (score > maxScore) {
        maxScore = score
      }
    }

    let sumExp = 0
    for (let row = 0; row < seqLen; row++) {
      const weight = Math.exp(scores[row]! - maxScore)
      scores[row] = weight
      sumExp += weight
    }

    const attendedData = new Float32Array(dim)
    for (let row = 0; row < seqLen; row++) {
      const weight = scores[row]! / sumExp
      const cacheOffset = row * dim
      for (let col = 0; col < dim; col++) {
        attendedData[col] += weight * cache.values[cacheOffset + col]
      }
    }

    const attended = T.make(1, dim, attendedData)
    return Ops.addSync(attended, input)
  }

  decodeStep(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.decodeStepSync(input, cache))
  }

  private static softmaxBackward(softmaxOutput: Tensor2D, gradOutput: Tensor2D): Tensor2D {
    const gradInput = T.zeros(softmaxOutput.rows, softmaxOutput.cols)
    SelfAttention.softmaxBackwardInto(softmaxOutput, gradOutput, gradInput)
    return gradInput
  }

  private static softmaxBackwardInto(softmaxOutput: Tensor2D, gradOutput: Tensor2D, gradInput: Tensor2D): void {
    for (let i = 0; i < softmaxOutput.rows; i++) {
      let dot = 0
      const rowOffset = i * softmaxOutput.cols
      for (let j = 0; j < softmaxOutput.cols; j++) {
        dot += softmaxOutput.data[rowOffset + j] * gradOutput.data[rowOffset + j]
      }
      for (let j = 0; j < softmaxOutput.cols; j++) {
        const y = softmaxOutput.data[rowOffset + j]
        const dy = gradOutput.data[rowOffset + j]
        gradInput.data[rowOffset + j] = y * (dy - dot)
      }
    }
  }

  backwardSync(dOut: Tensor2D, lr: number, cacheKey?: LayerCacheKey): Tensor2D {
    const cached = (cacheKey !== undefined ? this.cache.get(cacheKey) : undefined) ?? this.lastCache
    if (!cached) {
      throw new Ops.ShapeError("SelfAttention.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { input, q, k, v, attnWeights } = cached
    const scale = Math.sqrt(this.embeddingDim)
    const workspace = new TensorWorkspace()

    const gradAttnWeights = T.zeros(dOut.rows, v.rows)
    Ops.matMulIntoSync(dOut, v, gradAttnWeights, { transposeB: true, workspace })

    const gradV = T.zeros(attnWeights.cols, dOut.cols)
    Ops.matMulIntoSync(attnWeights, dOut, gradV, { transposeA: true, workspace })

    const gradScores = T.zeros(attnWeights.rows, attnWeights.cols)
    SelfAttention.softmaxBackwardInto(attnWeights, gradAttnWeights, gradScores)
    Ops.mulScalarInPlace(gradScores, 1 / scale)

    const gradQ = T.zeros(gradScores.rows, this.embeddingDim)
    Ops.matMulIntoSync(gradScores, k, gradQ, { workspace })
    const gradK = T.zeros(gradScores.cols, this.embeddingDim)
    Ops.matMulIntoSync(gradScores, q, gradK, { transposeA: true, workspace })

    const gradQKV = workspace.borrowTensor("gradQKV", gradQ.rows, this.embeddingDim * 3)
    this.packGradientsSync(gradQ, gradK, gradV, gradQKV)

    const gradWQKV = T.zeros(input.cols, this.embeddingDim * 3)
    Ops.matMulIntoSync(input, gradQKV, gradWQKV, { transposeA: true, workspace })

    const gradInputAttention = workspace.borrowTensor("gradInputAttention", gradQKV.rows, input.cols)
    Ops.matMulIntoSync(gradQKV, this.wQKV, gradInputAttention, { transposeB: true, workspace })

    const gradInput = T.zeros(gradInputAttention.rows, gradInputAttention.cols)
    Ops.addIntoSync(gradInputAttention, dOut, gradInput)

    this.projectionOptimizer.step(this.wQKV, gradWQKV, lr)

    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
