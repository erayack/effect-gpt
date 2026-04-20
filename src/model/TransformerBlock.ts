import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerCacheKey, LayerForwardContext, SyncModelLayer } from "./ModelLayer"
import { SelfAttention, type SelfAttentionKvCache } from "./SelfAttention"
import { FeedForward } from "./FeedForward"
import { LayerNorm } from "./LayerNorm"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import type { Rng } from "../tensor/random"

export interface TransformerBlockDecodeState {
  readonly attention: SelfAttentionKvCache
}

export class TransformerBlock implements SyncModelLayer {
  readonly _tag = "TransformerBlock"
  attention: SelfAttention
  feedForward: FeedForward
  norm1: LayerNorm
  norm2: LayerNorm

  constructor(embeddingDim: number = EMBEDDING_DIM, hiddenDim: number = HIDDEN_DIM, rng: Rng) {
    this.attention = new SelfAttention(embeddingDim, rng)
    this.feedForward = new FeedForward(embeddingDim, hiddenDim, rng)
    this.norm1 = new LayerNorm(embeddingDim)
    this.norm2 = new LayerNorm(embeddingDim)
  }

  get parametersCount(): number {
    return (
      this.attention.parametersCount +
      this.feedForward.parametersCount +
      this.norm1.parametersCount +
      this.norm2.parametersCount
    )
  }

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D {
    const attentionOut = this.attention.forwardSync(input, context)
    const norm1Out = this.norm1.forwardSync(attentionOut, context)
    const ffnOut = this.feedForward.forwardSync(norm1Out, context)
    return this.norm2.forwardSync(ffnOut, context)
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

  createDecodeState(capacity: number): TransformerBlockDecodeState {
    return {
      attention: this.attention.createKvCache(capacity)
    }
  }

  prefillSync(input: Tensor2D, state: TransformerBlockDecodeState): Tensor2D {
    const attentionOut = this.attention.prefillSync(input, state.attention)
    const norm1Out = this.norm1.forwardInferenceSync(attentionOut)
    const ffnOut = this.feedForward.forwardInferenceSync(norm1Out)
    return this.norm2.forwardInferenceSync(ffnOut)
  }

  prefill(input: Tensor2D, state: TransformerBlockDecodeState): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.prefillSync(input, state))
  }

  decodeStepSync(input: Tensor2D, state: TransformerBlockDecodeState): Tensor2D {
    const attentionOut = this.attention.decodeStepSync(input, state.attention)
    const norm1Out = this.norm1.forwardInferenceSync(attentionOut)
    const ffnOut = this.feedForward.forwardInferenceSync(norm1Out)
    return this.norm2.forwardInferenceSync(ffnOut)
  }

  decodeStep(input: Tensor2D, state: TransformerBlockDecodeState): Effect.Effect<Tensor2D, ShapeError> {
    return Ops.syncShapeEffect(() => this.decodeStepSync(input, state))
  }

  backwardSync(dOut: Tensor2D, lr: number, cacheKey?: LayerCacheKey): Tensor2D {
    let grad = this.norm2.backwardSync(dOut, lr, cacheKey)
    grad = this.feedForward.backwardSync(grad, lr, cacheKey)
    grad = this.norm1.backwardSync(grad, lr, cacheKey)
    grad = this.attention.backwardSync(grad, lr, cacheKey)
    return grad
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      return yield* Ops.syncShapeEffect(() => this.backwardSync(dOut, lr, this.fiberKey(fiberId)))
    })
  }
}
