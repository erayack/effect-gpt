import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { Adam } from "../training/Adam"

export class LayerNorm implements ModelLayer {
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

  forward(input: Tensor2D, _context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const mean = Ops.meanRows(input)
      const variance = Ops.varRows(input)
      const rstd = Ops.mapScalar(variance, (v) => 1.0 / Math.sqrt(v + this.epsilon))
      const centered = yield* Ops.broadcastSubCol(input, mean)
      const normalized = yield* Ops.broadcastMulCol(centered, rstd)

      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = {
        normalized,
        rstd
      }
      this.cache.set(key, cached)
      this.lastCache = cached

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

        // Gradient of LayerNorm: dL/dx = rstd * (dL/dnorm - mean(dL/dnorm) - norm * mean(dL/dnorm * norm))
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
    })
  }
}
