import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { Adam } from "../training/Adam"
import { TensorWorkspace } from "../tensor/Workspace"

export interface LayerNormInferenceState {
  readonly workspace: TensorWorkspace
}

interface LayerNormCacheEntry {
  readonly workspace: TensorWorkspace
  readonly normalized: Tensor2D
  readonly rstd: Tensor2D
}

export class LayerNorm implements SyncModelLayer {
  readonly _tag = "LayerNorm"
  readonly epsilon: number = 1e-5
  gamma: Tensor2D
  beta: Tensor2D

  private cache = new Map<LayerCacheKey, LayerNormCacheEntry>()
  private lastCache: LayerNormCacheEntry | null = null
  private readonly workspacePool: Array<TensorWorkspace> = []
  optimizerGamma: Adam
  optimizerBeta: Adam

  constructor(embeddingDim: number) {
    this.gamma = T.ones(1, embeddingDim)
    this.beta = T.zeros(1, embeddingDim)
    this.optimizerGamma = Adam.make(1, embeddingDim)
    this.optimizerBeta = Adam.make(1, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): LayerCacheKey {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.gamma.data.length + this.beta.data.length
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
    normalized: Tensor2D,
    rstd: Tensor2D
  ): void {
    if (cacheKey !== undefined && this.cache.has(cacheKey)) {
      throw new Ops.ShapeError("LayerNorm.forward received a cacheKey that is already active")
    }
    const cached = { workspace, normalized, rstd }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, cached)
    }
    this.lastCache = cached
  }

  private forwardCore(
    input: Tensor2D,
    captureCache: boolean,
    cacheKey?: LayerCacheKey,
    state?: LayerNormInferenceState
  ): Tensor2D {
    const workspace = captureCache ? this.acquireWorkspace() : state?.workspace ?? new TensorWorkspace()
    let stored = false
    try {
      const mean = workspace.borrowVector("mean", input.rows)
      const normalized = workspace.borrowTensor("normalized", input.rows, input.cols)
      const rstd = workspace.borrowTensor("rstd", input.rows, 1)

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

      const shifted = workspace.borrowTensor("shifted", input.rows, input.cols)
      for (let row = 0; row < input.rows; row++) {
        const rowOffset = row * input.cols
        for (let col = 0; col < input.cols; col++) {
          shifted.data[rowOffset + col] = normalized.data[rowOffset + col]! * this.gamma.data[col]! + this.beta.data[col]!
        }
      }
      if (captureCache) {
        this.storeCache(cacheKey, workspace, normalized, rstd)
        stored = true
      }
      return shifted
    } catch (error) {
      if (captureCache && !stored) {
        this.releaseWorkspace(workspace)
      }
      throw error
    }
  }

  forwardInferenceSync(input: Tensor2D, state?: LayerNormInferenceState): Tensor2D {
    return this.forwardCore(input, false, undefined, state)
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

    const { workspace, normalized, rstd } = cached
    try {
      const rows = normalized.rows
      const cols = normalized.cols
      const nFeatures = cols

      const gradNormalized = workspace.borrowTensor("gradNormalized", rows, cols)
      for (let i = 0; i < rows; i++) {
        const rowOffset = i * cols
        for (let j = 0; j < cols; j++) {
          const idx = rowOffset + j
          gradNormalized.data[idx] = this.gamma.data[j] * dOut.data[idx]
        }
      }

      const gradGamma = workspace.borrowTensor("gradGamma", 1, cols)
      const gradBeta = workspace.borrowTensor("gradBeta", 1, cols)
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

      const gradInput = workspace.borrowTensor("gradInput", rows, cols)
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
