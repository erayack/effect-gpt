import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { Adam } from "../training/Adam"
import { TensorWorkspace } from "../tensor/Workspace"

export class LayerNorm implements SyncModelLayer {
  readonly _tag = "LayerNorm"
  readonly epsilon: number = 1e-5
  gamma: Tensor2D
  beta: Tensor2D

  private cache = new Map<number | string, { normalized: Tensor2D; rstd: Tensor2D }>()
  private lastCache: { normalized: Tensor2D; rstd: Tensor2D } | null = null
  optimizerGamma: Adam
  optimizerBeta: Adam

  constructor(embeddingDim: number) {
    this.gamma = T.ones(1, embeddingDim)
    this.beta = T.zeros(1, embeddingDim)
    this.optimizerGamma = Adam.make(1, embeddingDim)
    this.optimizerBeta = Adam.make(1, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.gamma.data.length + this.beta.data.length
  }

  private storeCache(cacheKey: LayerCacheKey | undefined, normalized: Tensor2D, rstd: Tensor2D): void {
    const cached = { normalized, rstd }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, cached)
    }
    this.lastCache = cached
  }

  private forwardCore(input: Tensor2D, captureCache: boolean, cacheKey?: LayerCacheKey): Tensor2D {
    const workspace = new TensorWorkspace()
    const mean = workspace.borrowVector("mean", input.rows)
    const normalized = T.zeros(input.rows, input.cols)
    const rstd = T.zeros(input.rows, 1)

    for (let row = 0; row < input.rows; row++) {
      const rowOffset = row * input.cols
      let sum = 0
      for (let col = 0; col < input.cols; col++) {
        sum += input.data[rowOffset + col]!
      }
      mean[row] = sum / input.cols
    }

    for (let row = 0; row < input.rows; row++) {
      const rowOffset = row * input.cols
      let sumSq = 0
      const rowMean = mean[row]!
      for (let col = 0; col < input.cols; col++) {
        const diff = input.data[rowOffset + col]! - rowMean
        sumSq += diff * diff
      }
      const rowRstd = 1.0 / Math.sqrt(sumSq / input.cols + this.epsilon)
      rstd.data[row] = rowRstd
      for (let col = 0; col < input.cols; col++) {
        normalized.data[rowOffset + col] = (input.data[rowOffset + col]! - rowMean) * rowRstd
      }
    }

    if (captureCache) {
      this.storeCache(cacheKey, normalized, rstd)
    }

    const shifted = T.zeros(input.rows, input.cols)
    for (let row = 0; row < input.rows; row++) {
      const rowOffset = row * input.cols
      for (let col = 0; col < input.cols; col++) {
        shifted.data[rowOffset + col] = normalized.data[rowOffset + col]! * this.gamma.data[col]! + this.beta.data[col]!
      }
    }
    return shifted
  }

  forwardInferenceSync(input: Tensor2D): Tensor2D {
    return this.forwardCore(input, false)
  }

  forwardInference(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.forwardInferenceSync(input))
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    return this.forwardCore(input, context?.captureCache !== false, context?.cacheKey)
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
      throw new Ops.ShapeError("LayerNorm.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { normalized, rstd } = cached
    const rows = normalized.rows
    const cols = normalized.cols
    const nFeatures = cols

    const gradNormalized = T.zeros(rows, cols)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      for (let j = 0; j < cols; j++) {
        const idx = rowOffset + j
        gradNormalized.data[idx] = this.gamma.data[j] * dOut.data[idx]
      }
    }

    const gradGamma = T.zeros(1, cols)
    const gradBeta = T.zeros(1, cols)
    for (let j = 0; j < cols; j++) {
      let sumGamma = 0
      let sumBeta = 0
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j
        sumGamma += normalized.data[idx] * dOut.data[idx]
        sumBeta += dOut.data[idx]
      }
      gradGamma.data[j] = sumGamma
      gradBeta.data[j] = sumBeta
    }

    const gradInput = T.zeros(rows, cols)
    for (let i = 0; i < rows; i++) {
      const rstdVal = rstd.data[i]

      let sumGradNormalized = 0
      let sumGradNormTimesNorm = 0
      const rowOffset = i * cols

      for (let j = 0; j < cols; j++) {
        const idx = rowOffset + j
        sumGradNormalized += gradNormalized.data[idx]
        sumGradNormTimesNorm += gradNormalized.data[idx] * normalized.data[idx]
      }

      for (let j = 0; j < cols; j++) {
        const idx = rowOffset + j
        gradInput.data[idx] =
          rstdVal *
          (gradNormalized.data[idx] -
            sumGradNormalized / nFeatures -
            (normalized.data[idx] * sumGradNormTimesNorm) / nFeatures)
      }
    }

    this.optimizerGamma.step(this.gamma, gradGamma, lr)
    this.optimizerBeta.step(this.beta, gradBeta, lr)

    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
