import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"

export class FeedForward implements ModelLayer {
  readonly _tag = "FeedForward"
  w1: Tensor2D
  b1: Tensor2D
  w2: Tensor2D
  b2: Tensor2D

  private cache = new Map<
    number | string,
    { input: Tensor2D; hiddenPreActivation: Tensor2D; hiddenPostActivation: Tensor2D }
  >()
  private lastCache:
    | { input: Tensor2D; hiddenPreActivation: Tensor2D; hiddenPostActivation: Tensor2D }
    | null = null
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

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)

      const h1 = yield* Ops.matMul(input, this.w1)
      const h1Bias = yield* Ops.addRowBias(h1, this.b1)
      const h1BiasClone = T.clone(h1Bias)

      const h1Relu = Ops.relu(h1Bias)
      const h1ReluClone = T.clone(h1Relu)
      const cached = {
        input: T.clone(input),
        hiddenPreActivation: h1BiasClone,
        hiddenPostActivation: h1ReluClone
      }
      this.cache.set(key, cached)
      this.lastCache = cached

      const h2 = yield* Ops.matMul(h1Relu, this.w2)
      const h2Bias = yield* Ops.addRowBias(h2, this.b2)
      const output = yield* Ops.add(h2Bias, input)
      return output
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cached = this.cache.get(key) ?? this.lastCache
      if (!cached) {
        return yield* Effect.fail(new Ops.ShapeError("FeedForward.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const { input, hiddenPreActivation, hiddenPostActivation } = cached

      const hiddenPostT = Ops.transpose(hiddenPostActivation)
      const gradW2 = yield* Ops.matMul(hiddenPostT, dOut)
      const gradB2 = Ops.sumCols(dOut)

      const w2T = Ops.transpose(this.w2)
      const gradHiddenPost = yield* Ops.matMul(dOut, w2T)

      const reluGrad = T.zeros(hiddenPreActivation.rows, hiddenPreActivation.cols)
      for (let i = 0; i < hiddenPreActivation.data.length; i++) {
        reluGrad.data[i] = hiddenPreActivation.data[i] > 0 ? 1 : 0
      }
      const gradHiddenPre = yield* Ops.mul(gradHiddenPost, reluGrad)

      const inputT = Ops.transpose(input)
      const gradW1 = yield* Ops.matMul(inputT, gradHiddenPre)
      const gradB1 = Ops.sumCols(gradHiddenPre)

      const w1T = Ops.transpose(this.w1)
      const gradInputFF = yield* Ops.matMul(gradHiddenPre, w1T)
      const gradInput = yield* Ops.add(gradInputFF, dOut)

      this.optimizerW2.step(this.w2, gradW2, lr)
      this.optimizerB2.step(this.b2, gradB2, lr)
      this.optimizerW1.step(this.w1, gradW1, lr)
      this.optimizerB1.step(this.b1, gradB1, lr)

      return gradInput
    })
  }
}
