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

export interface OutputProjectionInferenceState {
  readonly workspace: TensorWorkspace
}

interface OutputProjectionCacheEntry {
  readonly workspace: TensorWorkspace
  readonly input: Tensor2D
}

export class OutputProjection implements SyncModelLayer {
  readonly _tag = "OutputProjection"
  wOut: Tensor2D
  bOut: Tensor2D

  private cache = new Map<LayerCacheKey, OutputProjectionCacheEntry>()
  private lastCache: OutputProjectionCacheEntry | null = null
  private readonly workspacePool: Array<TensorWorkspace> = []
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

  private acquireWorkspace(): TensorWorkspace {
    return this.workspacePool.pop() ?? new TensorWorkspace()
  }

  private releaseWorkspace(workspace: TensorWorkspace): void {
    this.workspacePool.push(workspace)
  }

  private storeCache(cacheKey: LayerCacheKey | undefined, workspace: TensorWorkspace, input: Tensor2D): void {
    if (cacheKey !== undefined && this.cache.has(cacheKey)) {
      throw new Ops.ShapeError("OutputProjection.forward received a cacheKey that is already active")
    }
    const entry = { workspace, input }
    if (cacheKey !== undefined) {
      this.cache.set(cacheKey, entry)
    }
    this.lastCache = entry
  }

  private forwardCore(
    input: Tensor2D,
    captureCache: boolean,
    cacheKey?: LayerCacheKey,
    state?: OutputProjectionInferenceState
  ): Tensor2D {
    const workspace = captureCache ? this.acquireWorkspace() : state?.workspace ?? new TensorWorkspace()
    let stored = false
    try {
      const projected = workspace.borrowTensor("projected", input.rows, this.wOut.cols)
      const output = workspace.borrowTensor("output", input.rows, this.wOut.cols)
      Ops.matMulIntoSync(input, this.wOut, projected, { workspace })
      Ops.addRowBiasIntoSync(projected, this.bOut, output)
      if (captureCache) {
        this.storeCache(cacheKey, workspace, input)
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

  forwardInferenceSync(input: Tensor2D, state?: OutputProjectionInferenceState): Tensor2D {
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
      throw new Ops.ShapeError("OutputProjection.backward called before forward")
    }
    if (cacheKey !== undefined) {
      this.cache.delete(cacheKey)
    }
    this.lastCache = null

    const { workspace, input } = cached
    try {
      const gradWOut = workspace.borrowTensor("gradWOut", input.cols, dOut.cols)
      Ops.matMulIntoSync(input, dOut, gradWOut, { transposeA: true, workspace })
      const gradBOut = workspace.borrowTensor("gradBOut", 1, dOut.cols)
      Ops.sumColsInto(dOut, gradBOut)

      const gradInput = workspace.borrowTensor("gradInput", dOut.rows, this.wOut.rows)
      Ops.matMulIntoSync(dOut, this.wOut, gradInput, { transposeB: true, workspace })

      this.optimizerWOut.step(this.wOut, gradWOut, lr)
      for (let j = 0; j < this.bOut.data.length; j++) {
        this.bOut.data[j] -= lr * gradBOut.data[j]
      }

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
