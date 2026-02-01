import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"

export class OutputProjection implements ModelLayer {
  readonly _tag = "OutputProjection"
  wOut: Tensor2D
  bOut: Tensor2D

  private cache = new Map<number | string, Tensor2D>()
  private lastCache: Tensor2D | null = null
  optimizerWOut: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, vocabSize: number, rng: Rng) {
    const std = Math.sqrt(2.0 / embeddingDim)
    this.wOut = Ops.initNormal(embeddingDim, vocabSize, 0, std, rng)
    this.bOut = T.zeros(1, vocabSize)
    this.optimizerWOut = Adam.make(embeddingDim, vocabSize)
  }

  private fiberKey(fiberId: FiberId.FiberId): number | string {
    return FiberId.isRuntime(fiberId) ? fiberId.id : JSON.stringify(fiberId)
  }

  get parametersCount(): number {
    return this.wOut.data.length + this.bOut.data.length
  }

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cloned = T.clone(input)
      this.cache.set(key, cloned)
      this.lastCache = cloned
      const projected = yield* Ops.matMul(input, this.wOut)
      const output = yield* Ops.addRowBias(projected, this.bOut)
      return output
    })
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      const cachedInput = this.cache.get(key) ?? this.lastCache
      if (!cachedInput) {
        return yield* Effect.fail(new Ops.ShapeError("OutputProjection.backward called before forward"))
      }
      this.cache.delete(key)
      this.lastCache = null

      const input = cachedInput
      const inputT = Ops.transpose(input)
      const gradWOut = yield* Ops.matMul(inputT, dOut)
      const gradBOut = Ops.sumCols(dOut)

      const wOutT = Ops.transpose(this.wOut)
      const gradInput = yield* Ops.matMul(dOut, wOutT)

      this.optimizerWOut.step(this.wOut, gradWOut, lr)
      for (let j = 0; j < this.bOut.data.length; j++) {
        this.bOut.data[j] -= lr * gradBOut.data[j]
      }

      return gradInput
    })
  }
}
