/**
 * Tests for Tier 2 test utilities.
 */
import { describe, expect, it } from "bun:test"
import * as T from "../../src/tensor/Tensor2D"
import {
  expectShape,
  expectClose,
  expectAllClose,
  expectNotClose,
  expectFinite
} from "./support/tensorMatchers"
import { runEffect, runEffectFail } from "./support/runEffect"
import {
  makeEmbeddings,
  makeSelfAttention,
  makeFeedForward,
  makeLayerNorm,
  makeTransformerBlock,
  makeOutputProjection,
  makeRng
} from "./support/factories"
import * as Effect from "effect/Effect"
import { ShapeError } from "../../src/tensor/ops"
import { EMBEDDING_DIM, HIDDEN_DIM } from "../../src/config"

describe("tensorMatchers", () => {
  it("expectShape passes for correct shape", () => {
    const t = T.zeros(3, 4)
    expectShape(t, [3, 4])
  })

  it("expectShape fails for wrong shape", () => {
    const t = T.zeros(3, 4)
    expect(() => expectShape(t, [3, 5])).toThrow()
  })

  it("expectClose passes for identical tensors", () => {
    const t1 = T.fromArray(2, 2, [1, 2, 3, 4])
    const t2 = T.fromArray(2, 2, [1, 2, 3, 4])
    expectClose(t1, t2)
  })

  it("expectClose passes within epsilon", () => {
    const t1 = T.fromArray(2, 2, [1.0, 2.0, 3.0, 4.0])
    const t2 = T.fromArray(2, 2, [1.000001, 2.000001, 3.000001, 4.000001])
    expectClose(t1, t2)
  })

  it("expectClose fails beyond epsilon", () => {
    const t1 = T.fromArray(2, 2, [1, 2, 3, 4])
    const t2 = T.fromArray(2, 2, [1, 2, 3, 5])
    expect(() => expectClose(t1, t2)).toThrow()
  })

  it("expectAllClose passes when all near value", () => {
    const t = T.fromArray(2, 2, [1.0, 1.000005, 0.999995, 1.0])
    expectAllClose(t, 1.0)
  })

  it("expectNotClose passes when tensors differ", () => {
    const t1 = T.fromArray(2, 2, [1, 2, 3, 4])
    const t2 = T.fromArray(2, 2, [1, 2, 3, 100])
    expectNotClose(t1, t2)
  })

  it("expectNotClose fails when tensors are equal", () => {
    const t1 = T.fromArray(2, 2, [1, 2, 3, 4])
    const t2 = T.fromArray(2, 2, [1, 2, 3, 4])
    expect(() => expectNotClose(t1, t2)).toThrow()
  })

  it("expectFinite passes for finite values", () => {
    const t = T.fromArray(2, 2, [1, 2, 3, 4])
    expectFinite(t)
  })

  it("expectFinite fails for NaN", () => {
    const t = T.fromArray(2, 2, [1, NaN, 3, 4])
    expect(() => expectFinite(t)).toThrow()
  })

  it("expectFinite fails for Infinity", () => {
    const t = T.fromArray(2, 2, [1, Infinity, 3, 4])
    expect(() => expectFinite(t)).toThrow()
  })
})

describe("runEffect", () => {
  it("returns value from successful effect", () => {
    const effect = Effect.succeed(42)
    expect(runEffect(effect)).toBe(42)
  })

  it("throws on failed effect", () => {
    const effect = Effect.fail(new ShapeError("test error"))
    expect(() => runEffect(effect)).toThrow()
  })

  it("runEffectFail returns the error", () => {
    const effect = Effect.fail(new ShapeError("test error"))
    const error = runEffectFail(effect)
    expect(error).toBeInstanceOf(ShapeError)
    expect(error.message).toBe("test error")
  })
})

describe("factories", () => {
  it("makeEmbeddings produces deterministic weights", () => {
    const e1 = makeEmbeddings(100, { seed: 42 })
    const e2 = makeEmbeddings(100, { seed: 42 })
    expectClose(e1.tokenEmbeddings, e2.tokenEmbeddings)
    expectClose(e1.positionalEmbeddings, e2.positionalEmbeddings)
  })

  it("makeEmbeddings with different seeds produces different weights", () => {
    const e1 = makeEmbeddings(100, { seed: 42 })
    const e2 = makeEmbeddings(100, { seed: 43 })
    expectNotClose(e1.tokenEmbeddings, e2.tokenEmbeddings)
  })

  it("makeSelfAttention produces deterministic weights", () => {
    const a1 = makeSelfAttention({ seed: 42 })
    const a2 = makeSelfAttention({ seed: 42 })
    expectClose(a1.wQ, a2.wQ)
    expectClose(a1.wK, a2.wK)
    expectClose(a1.wV, a2.wV)
  })

  it("makeFeedForward produces deterministic weights", () => {
    const f1 = makeFeedForward({ seed: 42 })
    const f2 = makeFeedForward({ seed: 42 })
    expectClose(f1.w1, f2.w1)
    expectClose(f1.w2, f2.w2)
  })

  it("makeLayerNorm initializes correctly", () => {
    const ln = makeLayerNorm()
    expectShape(ln.gamma, [1, EMBEDDING_DIM])
    expectShape(ln.beta, [1, EMBEDDING_DIM])
    expectAllClose(ln.gamma, 1.0)
    expectAllClose(ln.beta, 0.0)
  })

  it("makeTransformerBlock produces deterministic weights", () => {
    const tb1 = makeTransformerBlock({ seed: 42 })
    const tb2 = makeTransformerBlock({ seed: 42 })
    expectClose(tb1.attention.wQ, tb2.attention.wQ)
    expectClose(tb1.feedForward.w1, tb2.feedForward.w1)
  })

  it("makeOutputProjection produces deterministic weights", () => {
    const op1 = makeOutputProjection(100, { seed: 42 })
    const op2 = makeOutputProjection(100, { seed: 42 })
    expectClose(op1.wOut, op2.wOut)
  })

  it("makeRng produces deterministic sequences", () => {
    const rng1 = makeRng(42)
    const rng2 = makeRng(42)
    const seq1 = Array.from({ length: 5 }, () => rng1.next())
    const seq2 = Array.from({ length: 5 }, () => rng2.next())
    expect(seq1).toEqual(seq2)
  })
})
