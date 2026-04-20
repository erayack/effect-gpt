import { describe, test, expect } from "bun:test"
import * as Effect from "effect/Effect"
import type { Tensor2D } from "../../src/tensor/Tensor2D"
import * as T from "../../src/tensor/Tensor2D"
import type { ShapeError } from "../../src/tensor/ops"
import { EMBEDDING_DIM } from "../../src/config"
import {
  runBackwardPass,
  runForwardPass,
  type LayerForwardContext,
  type ModelLayer
} from "../../src/model/ModelLayer"
import { makeEmbeddings, makeOutputProjection } from "./support/factories"
import { expectShape, expectFinite } from "./support/tensorMatchers"

class EffectPassthroughLayer implements ModelLayer {
  readonly _tag = "EffectPassthrough"
  readonly parametersCount = 0

  forward(input: Tensor2D, _context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.sync(() => T.clone(input))
  }

  backward(dOut: Tensor2D, _lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.sync(() => T.clone(dOut))
  }
}

describe("ModelLayer runtime helpers", () => {
  test("mixed sync/effect networks preserve explicit cache keys across concurrent backward passes", async () => {
    const vocabSize = 8
    const network: ReadonlyArray<ModelLayer> = [
      makeEmbeddings(vocabSize),
      new EffectPassthroughLayer(),
      makeOutputProjection(vocabSize)
    ]

    const runStep = (input: Tensor2D, gradRows: number, cacheKey: symbol) =>
      Effect.gen(function* () {
        const logits = yield* runForwardPass(network, input, { cacheKey, captureCache: true })
        expectShape(logits, [gradRows, vocabSize])
        const grads = T.ones(gradRows, vocabSize)
        return yield* runBackwardPass(network, grads, 0.01, cacheKey)
      })

    const [gradA, gradB] = await Effect.runPromise(
      Effect.all(
        [
          runStep(T.fromArray(1, 2, [0, 1]), 2, Symbol("cache-a")),
          runStep(T.fromArray(1, 3, [2, 3, 4]), 3, Symbol("cache-b"))
        ],
        { concurrency: "unbounded" }
      )
    )

    expectShape(gradA, [2, EMBEDDING_DIM])
    expectShape(gradB, [3, EMBEDDING_DIM])
    expectFinite(gradA)
    expectFinite(gradB)
  })
})
