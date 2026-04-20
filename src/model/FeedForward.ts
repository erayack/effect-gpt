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

export class FeedForward implements SyncModelLayer {
  readonly _tag = "FeedForward"
  w1: Tensor2D
  b1: Tensor2D
  w2: Tensor2D
  b2: Tensor2D

  private cache = new Map<number | string, { input: Tensor2D; hiddenPostActivation: Tensor2D }>()
  private lastCache: { input: Tensor2D; hiddenPostActivation: Tensor2D } | null = null
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

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.w1.data.length + this.b1.data.length + this.w2.data.length + this.b2.data.length
  }

  private storeCache(cacheKey: LayerCacheKey | undefined, input: Tensor2D, hiddenPostActivation: Tensor2D): void {
    const cached = { input, hiddenPostActivation }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, cached)
    }
    this.lastCache = cached
  }

  private forwardCore(input: Tensor2D, captureCache: boolean, cacheKey?: LayerCacheKey): Tensor2D {
    const workspace = new TensorWorkspace()
    const hiddenPre = workspace.borrowTensor("hiddenPre", input.rows, this.w1.cols)
    const hiddenPost = T.zeros(input.rows, this.w1.cols)
    const outputPre = workspace.borrowTensor("outputPre", input.rows, this.w2.cols)
    const output = T.zeros(input.rows, input.cols)

    Ops.matMulIntoSync(input, this.w1, hiddenPre, { workspace })
    Ops.addRowBiasInPlaceSync(hiddenPre, this.b1)
    Ops.reluInto(hiddenPre, hiddenPost)
    if (captureCache) {
      this.storeCache(cacheKey, input, hiddenPost)
    }
    Ops.matMulIntoSync(hiddenPost, this.w2, outputPre, { workspace })
    Ops.addRowBiasInPlaceSync(outputPre, this.b2)
    Ops.addIntoSync(outputPre, input, output)
    return output
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
      throw new Ops.ShapeError("FeedForward.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { input, hiddenPostActivation } = cached
    const workspace = new TensorWorkspace()
    const gradW2 = T.zeros(hiddenPostActivation.cols, dOut.cols)
    Ops.matMulIntoSync(hiddenPostActivation, dOut, gradW2, { transposeA: true, workspace })
    const gradB2 = Ops.sumCols(dOut)

    const gradHiddenPost = workspace.borrowTensor("gradHiddenPost", dOut.rows, this.w2.rows)
    Ops.matMulIntoSync(dOut, this.w2, gradHiddenPost, { transposeB: true, workspace })

    const reluGrad = workspace.borrowTensor("reluGrad", hiddenPostActivation.rows, hiddenPostActivation.cols)
    for (let i = 0; i < hiddenPostActivation.data.length; i++) {
      reluGrad.data[i] = hiddenPostActivation.data[i] > 0 ? 1 : 0
    }
    const gradHiddenPre = T.zeros(gradHiddenPost.rows, gradHiddenPost.cols)
    Ops.mulIntoSync(gradHiddenPost, reluGrad, gradHiddenPre)

    const gradW1 = T.zeros(input.cols, gradHiddenPre.cols)
    Ops.matMulIntoSync(input, gradHiddenPre, gradW1, { transposeA: true, workspace })
    const gradB1 = Ops.sumCols(gradHiddenPre)

    const gradInputFF = workspace.borrowTensor("gradInputFF", gradHiddenPre.rows, this.w1.rows)
    Ops.matMulIntoSync(gradHiddenPre, this.w1, gradInputFF, { transposeB: true, workspace })
    const gradInput = T.zeros(gradInputFF.rows, gradInputFF.cols)
    Ops.addIntoSync(gradInputFF, dOut, gradInput)

    this.optimizerW2.step(this.w2, gradW2, lr)
    this.optimizerB2.step(this.b2, gradB2, lr)
    this.optimizerW1.step(this.w1, gradW1, lr)
    this.optimizerB1.step(this.b1, gradB1, lr)

    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
