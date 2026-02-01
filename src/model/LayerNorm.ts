import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { Adam } from "../training/Adam"

export class LayerNorm implements ModelLayer {
  readonly _tag = "LayerNorm"
  readonly epsilon: number = 1e-5
  gamma: Tensor2D
  beta: Tensor2D

  private cache = new Map<number | string, { input: Tensor2D; mean: Tensor2D; variance: Tensor2D }>()
  private lastCache: { input: Tensor2D; mean: Tensor2D; variance: Tensor2D } | null = null
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

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const mean = Ops.meanRows(input)
      const variance = Ops.varRows(input)

      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = {
        input: T.clone(input),
        mean: T.clone(mean),
        variance: T.clone(variance)
      }
      this.cache.set(key, cached)
      this.lastCache = cached

      // Use sqrt(variance + epsilon) for numerical stability
      const rstd = Ops.mapScalar(variance, (v) => 1.0 / Math.sqrt(v + this.epsilon))
      const centered = yield* Ops.broadcastSubCol(input, mean)
      const normalized = yield* Ops.broadcastMulCol(centered, rstd)
      const scaled = yield* Ops.broadcastMulRow(normalized, this.gamma)
      const shifted = yield* Ops.broadcastAddRow(scaled, this.beta)
      return shifted
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = this.cache.get(key) ?? this.lastCache
      if (!cached) {
        return yield* Effect.fail(new Ops.ShapeError("LayerNorm.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const { input, mean, variance } = cached
      const rows = input.rows
      const cols = input.cols
      const nFeatures = cols

      const normalized = T.zeros(rows, cols)
      const gradNormalized = T.zeros(rows, cols)
      for (let i = 0; i < rows; i++) {
        const meanVal = mean.data[i]
        // Consistent with forward: rstd = 1 / sqrt(variance + epsilon)
        const rstd = 1.0 / Math.sqrt(variance.data[i] + this.epsilon)
        for (let j = 0; j < cols; j++) {
          const idx = i * cols + j
          const norm = (input.data[idx] - meanVal) * rstd
          normalized.data[idx] = norm
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
        const meanVal = mean.data[i]
        const varPlusEps = variance.data[i] + this.epsilon
        const rstd = 1.0 / Math.sqrt(varPlusEps)

        let sumGradNormalized = 0
        let sumGradNormTimesNorm = 0
        const rowOffset = i * cols

        for (let j = 0; j < cols; j++) {
          const idx = rowOffset + j
          sumGradNormalized += gradNormalized.data[idx]
          sumGradNormTimesNorm += gradNormalized.data[idx] * normalized.data[idx]
        }

        // Gradient of LayerNorm: dL/dx = rstd * (dL/dnorm - mean(dL/dnorm) - norm * mean(dL/dnorm * norm))
        for (let j = 0; j < cols; j++) {
          const idx = rowOffset + j
          gradInput.data[idx] =
            rstd *
            (gradNormalized.data[idx] -
              sumGradNormalized / nFeatures -
              (normalized.data[idx] * sumGradNormTimesNorm) / nFeatures)
        }
      }

      this.optimizerGamma.step(this.gamma, gradGamma, lr)
      this.optimizerBeta.step(this.beta, gradBeta, lr)

      return gradInput
    })
  }
}
