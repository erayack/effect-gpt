import * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { SelfAttention } from "./SelfAttention"
import { FeedForward } from "./FeedForward"
import { LayerNorm } from "./LayerNorm"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import type { Rng } from "../tensor/random"

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

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const attentionOut: Tensor2D = yield* this.attention.forward(input)
      const norm1Out: Tensor2D = yield* this.norm1.forward(attentionOut)
      const ffnOut: Tensor2D = yield* this.feedForward.forward(norm1Out)
      const norm2Out: Tensor2D = yield* this.norm2.forward(ffnOut)
      return norm2Out
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
