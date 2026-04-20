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
  wQ: Tensor2D
  wK: Tensor2D
  wV: Tensor2D

  private cache = new Map<number | string, { input: Tensor2D; q: Tensor2D; k: Tensor2D; v: Tensor2D; attnWeights: Tensor2D }>()
  private lastCache: { input: Tensor2D; q: Tensor2D; k: Tensor2D; v: Tensor2D; attnWeights: Tensor2D } | null = null
  optimizerWQ: Adam
  optimizerWK: Adam
  optimizerWV: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, rng: Rng) {
    this.embeddingDim = embeddingDim
    const std = Math.sqrt(2.0 / embeddingDim)
    this.wQ = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.wK = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.wV = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.optimizerWQ = Adam.make(embeddingDim, embeddingDim)
    this.optimizerWK = Adam.make(embeddingDim, embeddingDim)
    this.optimizerWV = Adam.make(embeddingDim, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.wQ.data.length + this.wK.data.length + this.wV.data.length
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

  private computeQKVSync(input: Tensor2D): { q: Tensor2D; k: Tensor2D; v: Tensor2D } {
    const workspace = new TensorWorkspace()
    const q = T.zeros(input.rows, this.wQ.cols)
    const k = T.zeros(input.rows, this.wK.cols)
    const v = T.zeros(input.rows, this.wV.cols)
    Ops.matMulIntoSync(input, this.wQ, q, { workspace })
    Ops.matMulIntoSync(input, this.wK, k, { workspace })
    Ops.matMulIntoSync(input, this.wV, v, { workspace })
    return { q, k, v }
  }

  private attentionSync(
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    layout?: SequenceLayout
  ): { attnWeights: Tensor2D; attended: Tensor2D } {
    const dk = Math.sqrt(this.embeddingDim)
    const workspace = new TensorWorkspace()
    const scores = workspace.borrowTensor("scores", q.rows, k.rows)
    Ops.matMulIntoSync(q, k, scores, { transposeB: true, workspace })
    Ops.mulScalarInPlace(scores, 1 / dk)
    Ops.maskCausalInPlace(scores, layout)

    const attnWeights = T.zeros(scores.rows, scores.cols)
    Ops.softmaxRowsInto(scores, attnWeights)
    const attended = T.zeros(attnWeights.rows, v.cols)
    Ops.matMulIntoSync(attnWeights, v, attended, { workspace })
    return { attnWeights, attended }
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    const { q, k, v } = this.computeQKVSync(input)
    const { attnWeights, attended } = this.attentionSync(q, k, v, context?.sequenceLayout)
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
    const { q, k, v } = this.computeQKVSync(input)
    this.storeKvRowsSync(k, v, cache)
    const { attended } = this.attentionSync(q, k, v)
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

    const { q, k, v } = this.computeQKVSync(input)
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
    const scale = Math.sqrt(this.wQ.cols)
    const workspace = new TensorWorkspace()

    const gradAttnWeights = T.zeros(dOut.rows, v.rows)
    Ops.matMulIntoSync(dOut, v, gradAttnWeights, { transposeB: true, workspace })

    const gradV = T.zeros(attnWeights.cols, dOut.cols)
    Ops.matMulIntoSync(attnWeights, dOut, gradV, { transposeA: true, workspace })

    const gradScores = T.zeros(attnWeights.rows, attnWeights.cols)
    SelfAttention.softmaxBackwardInto(attnWeights, gradAttnWeights, gradScores)
    Ops.mulScalarInPlace(gradScores, 1 / scale)

    const gradQ = T.zeros(gradScores.rows, k.cols)
    Ops.matMulIntoSync(gradScores, k, gradQ, { workspace })
    const gradK = T.zeros(gradScores.cols, q.cols)
    Ops.matMulIntoSync(gradScores, q, gradK, { transposeA: true, workspace })

    const gradWQ = T.zeros(input.cols, gradQ.cols)
    const gradWK = T.zeros(input.cols, gradK.cols)
    const gradWV = T.zeros(input.cols, gradV.cols)
    Ops.matMulIntoSync(input, gradQ, gradWQ, { transposeA: true, workspace })
    Ops.matMulIntoSync(input, gradK, gradWK, { transposeA: true, workspace })
    Ops.matMulIntoSync(input, gradV, gradWV, { transposeA: true, workspace })

    const gradInputQ = workspace.borrowTensor("gradInputQ", gradQ.rows, this.wQ.rows)
    const gradInputK = workspace.borrowTensor("gradInputK", gradK.rows, this.wK.rows)
    const gradInputV = workspace.borrowTensor("gradInputV", gradV.rows, this.wV.rows)
    Ops.matMulIntoSync(gradQ, this.wQ, gradInputQ, { transposeB: true, workspace })
    Ops.matMulIntoSync(gradK, this.wK, gradInputK, { transposeB: true, workspace })
    Ops.matMulIntoSync(gradV, this.wV, gradInputV, { transposeB: true, workspace })

    const gradInputAttention = workspace.borrowTensor("gradInputAttention", gradInputQ.rows, gradInputQ.cols)
    Ops.addIntoSync(gradInputQ, gradInputK, gradInputAttention)
    const gradInputAttention2 = workspace.borrowTensor("gradInputAttention2", gradInputQ.rows, gradInputQ.cols)
    Ops.addIntoSync(gradInputAttention, gradInputV, gradInputAttention2)
    const gradInput = T.zeros(gradInputQ.rows, gradInputQ.cols)
    Ops.addIntoSync(gradInputAttention2, dOut, gradInput)

    this.optimizerWQ.step(this.wQ, gradWQ, lr)
    this.optimizerWK.step(this.wK, gradWK, lr)
    this.optimizerWV.step(this.wV, gradWV, lr)

    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
