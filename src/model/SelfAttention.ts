import * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"
import type { ShapeError } from "../tensor/ops"
import type { ModelLayer } from "./ModelLayer"
import { EMBEDDING_DIM } from "../config"
import { Adam } from "../training/Adam"
import type { Rng } from "../tensor/random"

export class SelfAttention implements ModelLayer {
  readonly _tag = "SelfAttention"
  readonly embeddingDim: number
  wQ: Tensor2D
  wK: Tensor2D
  wV: Tensor2D

  cachedInput: Tensor2D | null = null
  optimizerWQ: Adam
  optimizerWK: Adam
  optimizerWV: Adam

  constructor(embeddingDim: number = EMBEDDING_DIM, rng?: Rng) {
    this.embeddingDim = embeddingDim
    const std = Math.sqrt(2.0 / embeddingDim)
    this.wQ = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.wK = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.wV = Ops.initNormal(embeddingDim, embeddingDim, 0, std, rng)
    this.optimizerWQ = Adam.make(embeddingDim, embeddingDim)
    this.optimizerWK = Adam.make(embeddingDim, embeddingDim)
    this.optimizerWV = Adam.make(embeddingDim, embeddingDim)
  }

  get parametersCount(): number {
    return this.wQ.data.length + this.wK.data.length + this.wV.data.length
  }

  private computeQKV(input: Tensor2D): Effect.Effect<{ q: Tensor2D; k: Tensor2D; v: Tensor2D }, ShapeError> {
    return Effect.gen(this, function* () {
      const q: Tensor2D = yield* Ops.matMul(input, this.wQ)
      const k: Tensor2D = yield* Ops.matMul(input, this.wK)
      const v: Tensor2D = yield* Ops.matMul(input, this.wV)
      return { q, k, v }
    })
  }

  private attention(q: Tensor2D, k: Tensor2D, v: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      const dk = Math.sqrt(this.embeddingDim)
      const kT = Ops.transpose(k)
      const scores = yield* Ops.matMul(q, kT)
      const scaledScores = Ops.mulScalar(scores, 1 / dk)

      const seqLen = scaledScores.rows
      for (let i = 0; i < seqLen; i++) {
        for (let j = i + 1; j < seqLen; j++) {
          T.set(scaledScores, i, j, -Infinity)
        }
      }

      const weights = Ops.softmaxRows(scaledScores)
      const attended = yield* Ops.matMul(weights, v)
      return attended
    })
  }

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      this.cachedInput = T.clone(input)
      const { q, k, v } = yield* this.computeQKV(input)
      const attended = yield* this.attention(q, k, v)
      const output = yield* Ops.add(attended, input)
      return output
    })
  }

  private static softmaxBackward(softmaxOutput: Tensor2D, gradOutput: Tensor2D): Tensor2D {
    const gradInput = T.zeros(softmaxOutput.rows, softmaxOutput.cols)
    for (let i = 0; i < softmaxOutput.rows; i++) {
      let dot = 0
      const rowOffset = i * softmaxOutput.cols
      for (let j = 0; j < softmaxOutput.cols; j++) {
        dot += softmaxOutput.data[rowOffset + j] * gradOutput.data[rowOffset + j]
      }
      for (let j = 0; j < softmaxOutput.cols; j++) {
        const y = softmaxOutput.data[rowOffset + j]
        const dy = gradOutput.data[rowOffset + j]
        gradInput.data[rowOffset + j] = y * (dy - dot)
      }
    }
    return gradInput
  }

  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.gen(this, function* () {
      if (!this.cachedInput) {
        return yield* Effect.fail(new Ops.ShapeError("SelfAttention.backward called before forward"))
      }

      const input = this.cachedInput
      const q = yield* Ops.matMul(input, this.wQ)
      const k = yield* Ops.matMul(input, this.wK)
      const v = yield* Ops.matMul(input, this.wV)
      const scale = Math.sqrt(this.wQ.cols)

      const kT = Ops.transpose(k)
      const scores = yield* Ops.matMul(q, kT)
      const scaledScores = Ops.mulScalar(scores, 1 / scale)

      const seqLen = scaledScores.rows
      for (let i = 0; i < seqLen; i++) {
        for (let j = i + 1; j < seqLen; j++) {
          T.set(scaledScores, i, j, -Infinity)
        }
      }

      const attnWeights = Ops.softmaxRows(scaledScores)

      const vT = Ops.transpose(v)
      const gradAttnWeights = yield* Ops.matMul(dOut, vT)
      const attnWeightsT = Ops.transpose(attnWeights)
      const gradV = yield* Ops.matMul(attnWeightsT, dOut)

      const gradScores = SelfAttention.softmaxBackward(attnWeights, gradAttnWeights)
      const gradScoresScaled = Ops.mulScalar(gradScores, 1 / scale)

      const gradQ = yield* Ops.matMul(gradScoresScaled, k)
      const gradScoresScaledT = Ops.transpose(gradScoresScaled)
      const gradK = yield* Ops.matMul(gradScoresScaledT, q)

      const inputT = Ops.transpose(input)
      const gradWQ = yield* Ops.matMul(inputT, gradQ)
      const gradWK = yield* Ops.matMul(inputT, gradK)
      const gradWV = yield* Ops.matMul(inputT, gradV)

      const wQT = Ops.transpose(this.wQ)
      const wKT = Ops.transpose(this.wK)
      const wVT = Ops.transpose(this.wV)
      const gradInputQ = yield* Ops.matMul(gradQ, wQT)
      const gradInputK = yield* Ops.matMul(gradK, wKT)
      const gradInputV = yield* Ops.matMul(gradV, wVT)

      const gradInputAttention = yield* Ops.add(gradInputQ, gradInputK)
      const gradInputAttention2 = yield* Ops.add(gradInputAttention, gradInputV)
      const gradInput = yield* Ops.add(gradInputAttention2, dOut)

      this.optimizerWQ.step(this.wQ, gradWQ, lr)
      this.optimizerWK.step(this.wK, gradWK, lr)
      this.optimizerWV.step(this.wV, gradWV, lr)

      return gradInput
    })
  }
}
