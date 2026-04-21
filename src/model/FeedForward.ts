import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

export interface FeedForwardInferenceState {
  readonly workspace: TensorWorkspace
}

interface FeedForwardCacheEntry {
  readonly workspace: TensorWorkspace
  readonly input: Tensor2D
  readonly hiddenPostActivation: Tensor2D
}

export class FeedForward implements SyncModelLayer {
  readonly _tag = "FeedForward"
  w1: Tensor2D
  b1: Tensor2D
  w2: Tensor2D
  b2: Tensor2D

  private cache = new Map<LayerCacheKey, FeedForwardCacheEntry>()
  private lastCache: FeedForwardCacheEntry | null = null
  private readonly workspacePool: Array<TensorWorkspace> = []
  optimizerW1: Adam
  optimizerB1: Adam
  optimizerW2: Adam
  optimizerB2: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, hiddenDim: number = HIDDEN_DIM, rng: Rng) {
    const stdW1 = Math.sqrt(2.0 / embeddingDim)
    const stdW2 = Math.sqrt(2.0 / hiddenDim)

    this.w1 = Ops.initNormal(embeddingDim, hiddenDim, 0, stdW1, rng)
    this.b1 = T.zeros(1, hiddenDim)
    this.w2 = Ops.initNormal(hiddenDim, embeddingDim, 0, stdW2, rng)
    this.b2 = T.zeros(1, embeddingDim)
    this.optimizerW1 = Adam.make(embeddingDim, hiddenDim)
    this.optimizerB1 = Adam.make(1, hiddenDim)
    this.optimizerW2 = Adam.make(hiddenDim, embeddingDim)
    this.optimizerB2 = Adam.make(1, embeddingDim)
  }

  private fiberKey(fiberId: FiberId.FiberId): LayerCacheKey {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.w1.data.length + this.b1.data.length + this.w2.data.length + this.b2.data.length
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
    hiddenPostActivation: Tensor2D
  ): void {
    if (cacheKey !== undefined && this.cache.has(cacheKey)) {
      throw new Ops.ShapeError("FeedForward.forward received a cacheKey that is already active")
    }
    const cached = { workspace, input, hiddenPostActivation }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, cached)
    }
    this.lastCache = cached
  }

  private forwardCore(
    input: Tensor2D,
    captureCache: boolean,
    cacheKey?: LayerCacheKey,
    state?: FeedForwardInferenceState
  ): Tensor2D {
    const workspace = captureCache ? this.acquireWorkspace() : state?.workspace ?? new TensorWorkspace()
    let stored = false
    try {
      const hiddenPre = workspace.borrowTensor("hiddenPre", input.rows, this.w1.cols)
      const hiddenPost = workspace.borrowTensor("hiddenPost", input.rows, this.w1.cols)
      const outputPre = workspace.borrowTensor("outputPre", input.rows, this.w2.cols)
      const output = workspace.borrowTensor("output", input.rows, input.cols)

      Ops.matMulIntoSync(input, this.w1, hiddenPre, { workspace })
      Ops.addRowBiasInPlaceSync(hiddenPre, this.b1)
      Ops.reluInto(hiddenPre, hiddenPost)
      Ops.matMulIntoSync(hiddenPost, this.w2, outputPre, { workspace })
      Ops.addRowBiasInPlaceSync(outputPre, this.b2)
      Ops.addIntoSync(outputPre, input, output)
      if (captureCache) {
        this.storeCache(cacheKey, workspace, input, hiddenPost)
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

  forwardInferenceSync(input: Tensor2D, state?: FeedForwardInferenceState): Tensor2D {
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
      throw new Ops.ShapeError("FeedForward.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { workspace, input, hiddenPostActivation } = cached
    try {
      const gradW2 = workspace.borrowTensor("gradW2", hiddenPostActivation.cols, dOut.cols)
      Ops.matMulIntoSync(hiddenPostActivation, dOut, gradW2, { transposeA: true, workspace })
      const gradB2 = workspace.borrowTensor("gradB2", 1, dOut.cols)
      Ops.sumColsInto(dOut, gradB2)

      const gradHiddenPost = workspace.borrowTensor("gradHiddenPost", dOut.rows, this.w2.rows)
      Ops.matMulIntoSync(dOut, this.w2, gradHiddenPost, { transposeB: true, workspace })

      const reluGrad = workspace.borrowTensor("reluGrad", hiddenPostActivation.rows, hiddenPostActivation.cols)
      for (let i = 0; i < hiddenPostActivation.data.length; i++) {
        reluGrad.data[i] = hiddenPostActivation.data[i] > 0 ? 1 : 0
      }
      const gradHiddenPre = workspace.borrowTensor("gradHiddenPre", gradHiddenPost.rows, gradHiddenPost.cols)
      Ops.mulIntoSync(gradHiddenPost, reluGrad, gradHiddenPre)

      const gradW1 = workspace.borrowTensor("gradW1", input.cols, gradHiddenPre.cols)
      Ops.matMulIntoSync(input, gradHiddenPre, gradW1, { transposeA: true, workspace })
      const gradB1 = workspace.borrowTensor("gradB1", 1, gradHiddenPre.cols)
      Ops.sumColsInto(gradHiddenPre, gradB1)

      const gradInputFF = workspace.borrowTensor("gradInputFF", gradHiddenPre.rows, this.w1.rows)
      Ops.matMulIntoSync(gradHiddenPre, this.w1, gradInputFF, { transposeB: true, workspace })
      const gradInput = T.zeros(gradInputFF.rows, gradInputFF.cols)
      Ops.addIntoSync(gradInputFF, dOut, gradInput)

      this.optimizerW2.step(this.w2, gradW2, lr)
      this.optimizerB2.step(this.b2, gradB2, lr)
      this.optimizerW1.step(this.w1, gradW1, lr)
      this.optimizerB1.step(this.b1, gradB1, lr)

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
