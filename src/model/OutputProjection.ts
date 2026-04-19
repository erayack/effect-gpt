import * as Effect from "effect/Effect"
import * as FiberId from "effect/FiberId"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { LayerForwardContext, ModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"
import { TensorWorkspace } from "../tensor/Workspace"

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

  forwardInference(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const workspace = new TensorWorkspace()
      const projected = workspace.borrowTensor("projected", input.rows, this.wOut.cols)
      const output = T.zeros(input.rows, this.wOut.cols)
      yield* Ops.matMulInto(input, this.wOut, projected)
      yield* Ops.addRowBiasInto(projected, this.bOut, output)
      return output
    })
  }

  forward(input: Tensor2D, _context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const fiberId = yield* Effect.fiberId
      const key = this.fiberKey(fiberId)
      this.cache.set(key, input)
      this.lastCache = input
      const workspace = new TensorWorkspace()
      const projected = workspace.borrowTensor("projected", input.rows, this.wOut.cols)
      const output = T.zeros(input.rows, this.wOut.cols)
      yield* Ops.matMulInto(input, this.wOut, projected)
      yield* Ops.addRowBiasInto(projected, this.bOut, output)
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
      const workspace = new TensorWorkspace()
      const inputT = workspace.borrowTensor("inputT", input.cols, input.rows)
      Ops.transposeInto(input, inputT)
      const gradWOut = T.zeros(input.cols, dOut.cols)
      yield* Ops.matMulInto(inputT, dOut, gradWOut)
      const gradBOut = Ops.sumCols(dOut)

      const wOutT = workspace.borrowTensor("wOutT", this.wOut.cols, this.wOut.rows)
      Ops.transposeInto(this.wOut, wOutT)
      const gradInput = T.zeros(dOut.rows, this.wOut.rows)
      yield* Ops.matMulInto(dOut, wOutT, gradInput)

      this.optimizerWOut.step(this.wOut, gradWOut, lr)
      for (let j = 0; j < this.bOut.data.length; j++) {
        this.bOut.data[j] -= lr * gradBOut.data[j]
      }

      return gradInput
    })
  }
}
