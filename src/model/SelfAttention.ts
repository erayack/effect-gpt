import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer, SequenceLayout } from "./ModelLayer"
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

export class SelfAttention implements ModelLayer {
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

  private computeQKV(input: Tensor2D): Effect.Effect<{ q: Tensor2D; k: Tensor2D; v: Tensor2D }, ShapeError> {
    return Effect.gen(this, function* () {
      const workspace = new TensorWorkspace()
      const q = T.zeros(input.rows, this.wQ.cols)
      const k = T.zeros(input.rows, this.wK.cols)
      const v = T.zeros(input.rows, this.wV.cols)
      yield* Ops.matMulInto(input, this.wQ, q, { workspace })
      yield* Ops.matMulInto(input, this.wK, k, { workspace })
      yield* Ops.matMulInto(input, this.wV, v, { workspace })
      return { q, k, v }
    })
  }

  private attention(
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    layout?: SequenceLayout
  ): Effect.Effect<{ attnWeights: Tensor2D; attended: Tensor2D }, ShapeError> {
    return Effect.gen(this, function* () {
      const dk = Math.sqrt(this.embeddingDim)
      const workspace = new TensorWorkspace()
      const scores = workspace.borrowTensor("scores", q.rows, k.rows)
      yield* Ops.matMulInto(q, k, scores, { transposeB: true, workspace })
      Ops.mulScalarInPlace(scores, 1 / dk)
      Ops.maskCausalInPlace(scores, layout)

      const attnWeights = T.zeros(scores.rows, scores.cols)
      Ops.softmaxRowsInto(scores, attnWeights)
      const attended = T.zeros(attnWeights.rows, v.cols)
      yield* Ops.matMulInto(attnWeights, v, attended, { workspace })
      return { attnWeights, attended }
    })
  }

  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const { q, k, v } = yield* this.computeQKV(input)
      const { attnWeights, attended } = yield* this.attention(q, k, v, context?.sequenceLayout)
      const cached = { input, q, k, v, attnWeights }
      this.cache.set(key, cached)
      this.lastCache = cached
      const output = yield* Ops.add(attended, input)
      return output
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

  private validateKvCache(cache: SelfAttentionKvCache): Effect.Effect<void, ShapeError> {
    return Effect.sync(() => {
      if (cache.embeddingDim !== this.embeddingDim) {
        throw new Ops.ShapeError(
          `KV cache embedding dimension ${cache.embeddingDim} does not match attention dimension ${this.embeddingDim}`
        )
      }
      if (cache.length < 0 || cache.length > cache.capacity) {
        throw new Ops.ShapeError(`KV cache length ${cache.length} is out of bounds for capacity ${cache.capacity}`)
      }
    }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))
  }

  private ensureEmptyKvCache(cache: SelfAttentionKvCache): Effect.Effect<void, ShapeError> {
    return Effect.sync(() => {
      if (cache.length !== 0) {
        throw new Ops.ShapeError(
          `KV cache must be empty before prefill, received length ${cache.length}. Create a fresh cache for each prompt.`
        )
      }
    }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))
  }

  private storeKvRows(k: Tensor2D, v: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<void, ShapeError> {
    return Effect.sync(() => {
      const nextLength = cache.length + k.rows
      if (nextLength > cache.capacity) {
        throw new Ops.ShapeError(`KV cache capacity ${cache.capacity} exceeded by sequence length ${nextLength}`)
      }

      const offset = cache.length * this.embeddingDim
      cache.keys.set(k.data, offset)
      cache.values.set(v.data, offset)
      cache.length = nextLength
    }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))
  }

  prefill(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      yield* this.validateKvCache(cache)
      yield* this.ensureEmptyKvCache(cache)
      const { q, k, v } = yield* this.computeQKV(input)
      yield* this.storeKvRows(k, v, cache)
      const { attended } = yield* this.attention(q, k, v)
      return yield* Ops.add(attended, input)
    })
  }

  decodeStep(input: Tensor2D, cache: SelfAttentionKvCache): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      yield* this.validateKvCache(cache)
      if (input.rows !== 1) {
        return yield* Effect.fail(new Ops.ShapeError(`decodeStep expects a single row, received ${input.rows}`))
      }

      const { q, k, v } = yield* this.computeQKV(input)
      yield* this.storeKvRows(k, v, cache)

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
      return yield* Ops.add(attended, input)
    })
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

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = this.cache.get(key) ?? this.lastCache
      if (!cached) {
        return yield* Effect.fail(new Ops.ShapeError("SelfAttention.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const { input, q, k, v, attnWeights } = cached
      const scale = Math.sqrt(this.wQ.cols)
      const workspace = new TensorWorkspace()

      const gradAttnWeights = T.zeros(dOut.rows, v.rows)
      yield* Ops.matMulInto(dOut, v, gradAttnWeights, { transposeB: true, workspace })

      const gradV = T.zeros(attnWeights.cols, dOut.cols)
      yield* Ops.matMulInto(attnWeights, dOut, gradV, { transposeA: true, workspace })

      const gradScores = T.zeros(attnWeights.rows, attnWeights.cols)
      SelfAttention.softmaxBackwardInto(attnWeights, gradAttnWeights, gradScores)
      Ops.mulScalarInPlace(gradScores, 1 / scale)

      const gradQ = T.zeros(gradScores.rows, k.cols)
      yield* Ops.matMulInto(gradScores, k, gradQ, { workspace })
      const gradK = T.zeros(gradScores.cols, q.cols)
      yield* Ops.matMulInto(gradScores, q, gradK, { transposeA: true, workspace })

      const gradWQ = T.zeros(input.cols, gradQ.cols)
      const gradWK = T.zeros(input.cols, gradK.cols)
      const gradWV = T.zeros(input.cols, gradV.cols)
      yield* Ops.matMulInto(input, gradQ, gradWQ, { transposeA: true, workspace })
      yield* Ops.matMulInto(input, gradK, gradWK, { transposeA: true, workspace })
      yield* Ops.matMulInto(input, gradV, gradWV, { transposeA: true, workspace })

      const gradInputQ = workspace.borrowTensor("gradInputQ", gradQ.rows, this.wQ.rows)
      const gradInputK = workspace.borrowTensor("gradInputK", gradK.rows, this.wK.rows)
      const gradInputV = workspace.borrowTensor("gradInputV", gradV.rows, this.wV.rows)
      yield* Ops.matMulInto(gradQ, this.wQ, gradInputQ, { transposeB: true, workspace })
      yield* Ops.matMulInto(gradK, this.wK, gradInputK, { transposeB: true, workspace })
      yield* Ops.matMulInto(gradV, this.wV, gradInputV, { transposeB: true, workspace })

      const gradInputAttention = workspace.borrowTensor("gradInputAttention", gradInputQ.rows, gradInputQ.cols)
      yield* Ops.addInto(gradInputQ, gradInputK, gradInputAttention)
      const gradInputAttention2 = workspace.borrowTensor("gradInputAttention2", gradInputQ.rows, gradInputQ.cols)
      yield* Ops.addInto(gradInputAttention, gradInputV, gradInputAttention2)
      const gradInput = T.zeros(gradInputQ.rows, gradInputQ.cols)
      yield* Ops.addInto(gradInputAttention2, dOut, gradInput)

      this.optimizerWQ.step(this.wQ, gradWQ, lr)
      this.optimizerWK.step(this.wK, gradWK, lr)
      this.optimizerWV.step(this.wV, gradWV, lr)

      return gradInput
    })
  }
}
