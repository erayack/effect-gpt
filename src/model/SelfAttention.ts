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

interface SelfAttentionCacheEntry {
  readonly workspace: TensorWorkspace
  readonly input: Tensor2D
  readonly q: Tensor2D
  readonly k: Tensor2D
  readonly v: Tensor2D
  readonly sequenceLayout?: SequenceLayout
}

export class SelfAttention implements SyncModelLayer {
  readonly _tag = "SelfAttention"
  readonly embeddingDim: number
  wQKV: Tensor2D

  private cache = new Map<LayerCacheKey, SelfAttentionCacheEntry>()
  private lastCache: SelfAttentionCacheEntry | null = null
  private readonly workspacePool: Array<TensorWorkspace> = []
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

  private acquireWorkspace(): TensorWorkspace {
    return this.workspacePool.pop() ?? new TensorWorkspace()
  }

  private releaseWorkspace(workspace: TensorWorkspace): void {
    this.workspacePool.push(workspace)
  }

  private storeCache(
    cacheKey: LayerCacheKey | undefined,
    workspace: TensorWorkspace,
    input: Tensor2D,
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    sequenceLayout?: SequenceLayout
  ): void {
    if (cacheKey !== undefined && this.cache.has(cacheKey)) {
      throw new Ops.ShapeError("SelfAttention.forward received a cacheKey that is already active")
    }
    const cached: SelfAttentionCacheEntry = {
      workspace,
      input,
      q,
      k,
      v,
      ...(sequenceLayout ? { sequenceLayout } : {})
    }
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
    options?: {
      readonly layout?: SequenceLayout
      readonly causalMask?: boolean
      readonly weightsOut?: Tensor2D
    }
  ): Tensor2D {
    const attended = workspace.borrowTensor("attended", q.rows, v.cols)
    const fusedOptions: Ops.FusedScaledDotProductAttentionOptions = {
      causalMask: options?.causalMask ?? true,
      workspace,
      ...(options?.layout ? { layout: options.layout } : {}),
      ...(options?.weightsOut ? { weightsOut: options.weightsOut } : {})
    }
    Ops.fusedScaledDotProductAttentionIntoSync(q, k, v, attended, fusedOptions)
    return attended
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    const captureCache = context?.captureCache !== false
    const workspace = captureCache ? this.acquireWorkspace() : new TensorWorkspace()
    let stored = false
    try {
      const { q, k, v } = this.computeQKVSync(input, workspace)
      const attended = this.attentionSync(q, k, v, workspace, {
        causalMask: true,
        ...(context?.sequenceLayout ? { layout: context.sequenceLayout } : {})
      })
      const output = workspace.borrowTensor("output", attended.rows, attended.cols)
      Ops.addIntoSync(attended, input, output)
      if (captureCache) {
        this.storeCache(context?.cacheKey, workspace, input, q, k, v, context?.sequenceLayout)
        stored = true
      }
      return output
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

  prefillSync(input: Tensor2D, cache: SelfAttentionKvCache, workspace: TensorWorkspace = new TensorWorkspace()): Tensor2D {
    this.validateKvCacheSync(cache)
    this.ensureEmptyKvCacheSync(cache)
    const { q, k, v } = this.computeQKVSync(input, workspace)
    this.storeKvRowsSync(k, v, cache)
    const attended = this.attentionSync(q, k, v, workspace, { causalMask: true })
    const output = workspace.borrowTensor("output", attended.rows, attended.cols)
    Ops.addIntoSync(attended, input, output)
    return output
  }

  prefill(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.prefillSync(input, cache))
  }

  decodeStepSync(input: Tensor2D, cache: SelfAttentionKvCache, workspace: TensorWorkspace = new TensorWorkspace()): Tensor2D {
    this.validateKvCacheSync(cache)
    if (input.rows !== 1) {
      throw new Ops.ShapeError(`decodeStep expects a single row, received ${input.rows}`)
    }

    const { q, k, v } = this.computeQKVSync(input, workspace)
    this.storeKvRowsSync(k, v, cache)

    const seqLen = cache.length
    const dim = this.embeddingDim
    const scale = Math.sqrt(dim)
    const scores = workspace.borrowVectorAtLeast("scores", seqLen)
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

    const attended = workspace.borrowTensor("attended", 1, dim)
    attended.data.fill(0)
    for (let row = 0; row < seqLen; row++) {
      const weight = scores[row]! / sumExp
      const cacheOffset = row * dim
      for (let col = 0; col < dim; col++) {
        attended.data[col] += weight * cache.values[cacheOffset + col]
      }
    }

    const output = workspace.borrowTensor("output", 1, dim)
    Ops.addIntoSync(attended, input, output)
    return output
  }

  decodeStep(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.decodeStepSync(input, cache))
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

    const { workspace, input, q, k, v, sequenceLayout } = cached
    try {
      const gradQKVScratch = workspace.borrowVectorAtLeast(
        "fusedSdpaGradQKV",
        q.rows * this.embeddingDim + k.rows * this.embeddingDim + v.rows * v.cols
      )
      const gradQSize = q.rows * this.embeddingDim
      const gradKSize = k.rows * this.embeddingDim
      const gradVSize = v.rows * v.cols
      const gradQ = T.make(q.rows, this.embeddingDim, gradQKVScratch.subarray(0, gradQSize))
      const gradK = T.make(k.rows, this.embeddingDim, gradQKVScratch.subarray(gradQSize, gradQSize + gradKSize))
      const gradV = T.make(v.rows, v.cols, gradQKVScratch.subarray(gradQSize + gradKSize, gradQSize + gradKSize + gradVSize))
      Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, gradQ, gradK, gradV, {
        causalMask: true,
        workspace,
        ...(sequenceLayout ? { layout: sequenceLayout } : {})
      })

      const gradQKV = workspace.borrowTensor("gradQKV", gradQ.rows, this.embeddingDim * 3)
      this.packGradientsSync(gradQ, gradK, gradV, gradQKV)

      const gradWQKV = workspace.borrowTensor("gradWQKV", input.cols, this.embeddingDim * 3)
      Ops.matMulIntoSync(input, gradQKV, gradWQKV, { transposeA: true, workspace })

      const gradInputAttention = workspace.borrowTensor("gradInputAttention", gradQKV.rows, input.cols)
      Ops.matMulIntoSync(gradQKV, this.wQKV, gradInputAttention, { transposeB: true, workspace })

      const gradInput = T.zeros(gradInputAttention.rows, gradInputAttention.cols)
      Ops.addIntoSync(gradInputAttention, dOut, gradInput)

      this.projectionOptimizer.step(this.wQKV, gradWQKV, lr)

      return gradInput
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
