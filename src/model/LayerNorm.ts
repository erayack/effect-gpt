import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { Adam } from "../training/Adam"
import { TensorWorkspace } from "../tensor/Workspace"

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

  forwardInference(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const workspace = new TensorWorkspace()
      const mean = workspace.borrowVector("mean", input.rows)
      const rstd = workspace.borrowVector("rstd", input.rows)
      const output = T.zeros(input.rows, input.cols)

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
        for (let col = 0; col < input.cols; col++) {
          const diff = input.data[rowOffset + col]! - mean[row]!
          sumSq += diff * diff
        }
        rstd[row] = 1.0 / Math.sqrt(sumSq / input.cols + this.epsilon)
      }

      for (let row = 0; row < input.rows; row++) {
        const rowOffset = row * input.cols
        const rowRstd = rstd[row]!
        const rowMean = mean[row]!
        for (let col = 0; col < input.cols; col++) {
          const normalizedValue = (input.data[rowOffset + col]! - rowMean) * rowRstd
          output.data[rowOffset + col] = normalizedValue * this.gamma.data[col]! + this.beta.data[col]!
        }
      }

      return output
    })
  }

  forward(input: Tensor2D, _context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
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

      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = {
        normalized,
        rstd
      }
      this.cache.set(key, cached)
      this.lastCache = cached

      const shifted = T.zeros(input.rows, input.cols)
      for (let row = 0; row < input.rows; row++) {
        const rowOffset = row * input.cols
        for (let col = 0; col < input.cols; col++) {
          shifted.data[rowOffset + col] = normalized.data[rowOffset + col]! * this.gamma.data[col]! + this.beta.data[col]!
        }
      }
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
