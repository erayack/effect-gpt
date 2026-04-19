import * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { SelfAttention, type SelfAttentionKvCache } from "./SelfAttention"
import { FeedForward } from "./FeedForward"
import { LayerNorm } from "./LayerNorm"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import type { Rng } from "../tensor/random"

export interface TransformerBlockDecodeState {
  readonly attention: SelfAttentionKvCache
}

export class TransformerBlock implements ModelLayer {
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

  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const attentionOut: Tensor2D = yield* this.attention.forward(input, context)
      const norm1Out: Tensor2D = yield* this.norm1.forward(attentionOut, context)
      const ffnOut: Tensor2D = yield* this.feedForward.forward(norm1Out, context)
      const norm2Out: Tensor2D = yield* this.norm2.forward(ffnOut, context)
      return norm2Out
    })
  }

  createDecodeState(capacity: number): TransformerBlockDecodeState {
    return {
      attention: this.attention.createKvCache(capacity)
    }
  }

  prefill(input: Tensor2D, state: TransformerBlockDecodeState): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const attentionOut: Tensor2D = yield* this.attention.prefill(input, state.attention)
      const norm1Out: Tensor2D = yield* this.norm1.forwardInference(attentionOut)
      const ffnOut: Tensor2D = yield* this.feedForward.forwardInference(norm1Out)
      return yield* this.norm2.forwardInference(ffnOut)
    })
  }

  decodeStep(input: Tensor2D, state: TransformerBlockDecodeState): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const attentionOut: Tensor2D = yield* this.attention.decodeStep(input, state.attention)
      const norm1Out: Tensor2D = yield* this.norm1.forwardInference(attentionOut)
      const ffnOut: Tensor2D = yield* this.feedForward.forwardInference(norm1Out)
      return yield* this.norm2.forwardInference(ffnOut)
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      let grad: Tensor2D = yield* this.norm2.backward(dOut, lr)
      grad = yield* this.feedForward.backward(grad, lr)
      grad = yield* this.norm1.backward(grad, lr)
      grad = yield* this.attention.backward(grad, lr)
      return grad
    })
  }
}
