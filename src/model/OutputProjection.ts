import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

export class OutputProjection implements SyncModelLayer {
  readonly _tag = "OutputProjection"
  wOut: Tensor2D
  bOut: Tensor2D

  private cache = new Map<LayerCacheKey, Tensor2D>()
  private lastCache: Tensor2D | null = null
  optimizerWOut: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, vocabSize: number, rng: Rng) {
    const std = Math.sqrt(2.0 / embeddingDim)
    this.wOut = Ops.initNormal(embeddingDim, vocabSize, 0, std, rng)
    this.bOut = T.zeros(1, vocabSize)
    this.optimizerWOut = Adam.make(embeddingDim, vocabSize)
  }

  private fiberKey(fiberId: FiberId.FiberId): LayerCacheKey {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.wOut.data.length + this.bOut.data.length
  }

  private storeCache(cacheKey: LayerCacheKey | undefined, input: Tensor2D): void {
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, input)
    }
    this.lastCache = input
  }

  private forwardCore(input: Tensor2D, captureCache: boolean, cacheKey?: LayerCacheKey): Tensor2D {
    if (captureCache) {
      this.storeCache(cacheKey, input)
    }
    const workspace = new TensorWorkspace()
    const projected = workspace.borrowTensor("projected", input.rows, this.wOut.cols)
    const output = T.zeros(input.rows, this.wOut.cols)
    Ops.matMulIntoSync(input, this.wOut, projected, { workspace })
    Ops.addRowBiasIntoSync(projected, this.bOut, output)
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
    const cachedInput = (cacheKey !== undefined ? this.cache.get(cacheKey) : undefined) ?? this.lastCache
    if (!cachedInput) {
      throw new Ops.ShapeError("OutputProjection.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const input = cachedInput
    const workspace = new TensorWorkspace()
    const gradWOut = T.zeros(input.cols, dOut.cols)
    Ops.matMulIntoSync(input, dOut, gradWOut, { transposeA: true, workspace })
    const gradBOut = Ops.sumCols(dOut)

    const gradInput = T.zeros(dOut.rows, this.wOut.rows)
    Ops.matMulIntoSync(dOut, this.wOut, gradInput, { transposeB: true, workspace })

    this.optimizerWOut.step(this.wOut, gradWOut, lr)
    for (let j = 0; j < this.bOut.data.length; j++) {
      this.bOut.data[j] -= lr * gradBOut.data[j]
    }

    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
