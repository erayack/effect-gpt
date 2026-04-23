import { describe, test, expect } from "bun:test"
import { runEffect } from "./support/runEffect"
import { runEffectFail } from "./support/runEffect"
import * as T from "../../src/tensor/Tensor2D"
import * as Ops from "../../src/tensor/ops"
import { crossEntropyLossAndDLogits, crossEntropyLossAndDLogitsFromLogits } from "../../src/training/loss"
import { expectClose } from "./support/tensorMatchers"
import { TensorWorkspace } from "../../src/tensor/Workspace"

const makePatternedTensor = (rows: number, cols: number, offset: number): T.Tensor2D => {
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i + offset) % 17) / 5 - 1.5
  }
  return T.make(rows, cols, data)
}

const referenceMatMul = (
  a: T.Tensor2D,
  b: T.Tensor2D,
  options?: { transposeA?: boolean; transposeB?: boolean }
): T.Tensor2D => {
  const transposeA = options?.transposeA === true
  const transposeB = options?.transposeB === true
  const rows = transposeA ? a.cols : a.rows
  const sharedDim = transposeA ? a.rows : a.cols
  const cols = transposeB ? b.rows : b.cols
  const out = T.zeros(rows, cols)

  for (let row = 0; row < rows; row++) {
    const outRowOffset = row * cols
    for (let col = 0; col < cols; col++) {
      let sum = 0
      for (let k = 0; k < sharedDim; k++) {
        const aValue = transposeA ? a.data[k * a.cols + row]! : a.data[row * a.cols + k]!
        const bValue = transposeB ? b.data[col * b.cols + k]! : b.data[k * b.cols + col]!
        sum += aValue * bValue
      }
      out.data[outRowOffset + col] = sum
    }
  }

  return out
}

const referenceCausalAttention = (
  q: T.Tensor2D,
  k: T.Tensor2D,
  v: T.Tensor2D,
  layout?: { sequenceIds: Int32Array; positionIds: Int32Array; totalTokens: number }
): { attended: T.Tensor2D; weights: T.Tensor2D } => {
  const scores = referenceMatMul(q, k, { transposeB: true })
  Ops.mulScalarInPlace(scores, 1 / Math.sqrt(q.cols))
  Ops.maskCausalInPlace(scores, layout)
  const weights = Ops.softmaxRows(scores)
  const attended = referenceMatMul(weights, v)
  return { attended, weights }
}

const referenceSdpaBackward = (
  q: T.Tensor2D,
  k: T.Tensor2D,
  v: T.Tensor2D,
  dOut: T.Tensor2D,
  layout?: { sequenceIds: Int32Array; positionIds: Int32Array; totalTokens: number }
): { dQ: T.Tensor2D; dK: T.Tensor2D; dV: T.Tensor2D } => {
  const { weights } = referenceCausalAttention(q, k, v, layout)
  const gradAttnWeights = referenceMatMul(dOut, v, { transposeB: true })
  const dV = referenceMatMul(weights, dOut, { transposeA: true })
  const gradScores = T.zeros(weights.rows, weights.cols)
  const scale = 1 / Math.sqrt(q.cols)

  for (let row = 0; row < weights.rows; row++) {
    const rowOffset = row * weights.cols
    let dot = 0
    for (let col = 0; col < weights.cols; col++) {
      dot += weights.data[rowOffset + col]! * gradAttnWeights.data[rowOffset + col]!
    }
    for (let col = 0; col < weights.cols; col++) {
      const weight = weights.data[rowOffset + col]!
      const gradWeight = gradAttnWeights.data[rowOffset + col]!
      gradScores.data[rowOffset + col] = weight * (gradWeight - dot) * scale
    }
  }

  const dQ = referenceMatMul(gradScores, k)
  const dK = referenceMatMul(gradScores, q, { transposeA: true })
  return { dQ, dK, dV }
}

describe("buffered tensor ops", () => {
  test("matMulInto matches allocating matMul", () => {
    const a = T.fromArray(2, 3, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(3, 2, [7, 8, 9, 10, 11, 12])

    const expected = runEffect(Ops.matMul(a, b))
    const actual = T.zeros(2, 2)
    runEffect(Ops.matMulInto(a, b, actual))

    expectClose(actual, expected)
  })

  test("matMulInto supports transposeA", () => {
    const a = T.fromArray(3, 2, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(3, 2, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)

    runEffect(Ops.matMulInto(a, b, actual, { transposeA: true }))

    expectClose(actual, T.fromArray(2, 2, [89, 98, 116, 128]))
  })

  test("matMulInto supports transposeB with workspace reuse", () => {
    const a = T.fromArray(2, 3, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(2, 3, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)
    const workspace = new TensorWorkspace()

    runEffect(Ops.matMulInto(a, b, actual, { transposeB: true, workspace }))

    expectClose(actual, T.fromArray(2, 2, [50, 68, 122, 167]))
  })

  test("TensorWorkspace reuses backing storage for smaller tensor views", () => {
    const workspace = new TensorWorkspace()

    const large = workspace.borrowTensor("buffer", 4, 4)
    large.data.fill(1)
    const small = workspace.borrowTensor("buffer", 2, 2)

    expect(small.data.buffer).toBe(large.data.buffer)
    expect(small.rows).toBe(2)
    expect(small.cols).toBe(2)
  })

  test("TensorWorkspace grows once and reuses the larger backing storage", () => {
    const workspace = new TensorWorkspace()

    const small = workspace.borrowTensor("buffer", 2, 2)
    const grown = workspace.borrowTensor("buffer", 4, 4)
    const reused = workspace.borrowTensor("buffer", 3, 3)

    expect(grown.data.buffer).not.toBe(small.data.buffer)
    expect(reused.data.buffer).toBe(grown.data.buffer)
  })

  test("matMulInto supports transposeA and transposeB together", () => {
    const a = T.fromArray(3, 2, [1, 2, 3, 4, 5, 6])
    const b = T.fromArray(2, 3, [7, 8, 9, 10, 11, 12])
    const actual = T.zeros(2, 2)

    runEffect(Ops.matMulInto(a, b, actual, { transposeA: true, transposeB: true }))

    expectClose(actual, T.fromArray(2, 2, [76, 103, 100, 136]))
  })

  test("matMulInto rejects aliasing output storage", () => {
    const a = T.fromArray(2, 2, [1, 2, 3, 4])
    const b = T.fromArray(2, 2, [5, 6, 7, 8])

    const error = runEffectFail(Ops.matMulInto(a, b, a))

    expect(error.message).toContain("must not alias input storage")
  })

  test("blocked matMulInto matches reference implementation across transpose modes", () => {
    const nnA = makePatternedTensor(24, 48, 1)
    const nnB = makePatternedTensor(48, 40, 2)
    const nnActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(nnA, nnB, nnActual))
    expectClose(nnActual, referenceMatMul(nnA, nnB))

    const ntA = makePatternedTensor(24, 48, 3)
    const ntB = makePatternedTensor(40, 48, 4)
    const ntActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(ntA, ntB, ntActual, { transposeB: true }))
    expectClose(ntActual, referenceMatMul(ntA, ntB, { transposeB: true }))

    const tnA = makePatternedTensor(48, 24, 5)
    const tnB = makePatternedTensor(48, 40, 6)
    const tnActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(tnA, tnB, tnActual, { transposeA: true }))
    expectClose(tnActual, referenceMatMul(tnA, tnB, { transposeA: true }))

    const ttA = makePatternedTensor(48, 24, 7)
    const ttB = makePatternedTensor(40, 48, 8)
    const ttActual = T.zeros(24, 40)
    runEffect(Ops.matMulInto(ttA, ttB, ttActual, { transposeA: true, transposeB: true }))
    expectClose(ttActual, referenceMatMul(ttA, ttB, { transposeA: true, transposeB: true }))
  })

  test("softmaxRowsInto matches allocating softmaxRows", () => {
    const logits = T.fromArray(2, 4, [1, 2, 3, 4, 4, 3, 2, 1])
    const expected = Ops.softmaxRows(logits)
    const actual = T.zeros(2, 4)

    Ops.softmaxRowsInto(logits, actual)

    expectClose(actual, expected)
  })

  test("softmaxRowsInPlace normalizes each row", () => {
    const logits = T.fromArray(2, 3, [1, 2, 3, -1, 0, 1])

    Ops.softmaxRowsInPlace(logits)

    for (let row = 0; row < logits.rows; row++) {
      const rowOffset = row * logits.cols
      let sum = 0
      for (let col = 0; col < logits.cols; col++) {
        sum += logits.data[rowOffset + col]!
      }
      expect(sum).toBeCloseTo(1, 6)
    }
  })

  test("fused scaled-dot-product attention matches composed causal path", () => {
    const q = makePatternedTensor(4, 6, 9)
    const k = makePatternedTensor(4, 6, 10)
    const v = makePatternedTensor(4, 6, 11)
    const workspace = new TensorWorkspace()
    const actual = T.zeros(4, 6)
    const actualWeights = T.zeros(4, 4)

    const expected = referenceCausalAttention(q, k, v)

    Ops.fusedScaledDotProductAttentionIntoSync(q, k, v, actual, {
      causalMask: true,
      weightsOut: actualWeights,
      workspace
    })

    expectClose(actual, expected.attended)
    expectClose(actualWeights, expected.weights)
  })

  test("fused scaled-dot-product attention respects sequence layout masks", () => {
    const q = makePatternedTensor(4, 5, 12)
    const k = makePatternedTensor(4, 5, 13)
    const v = makePatternedTensor(4, 5, 14)
    const layout = {
      totalTokens: 4,
      sequenceIds: new Int32Array([0, 0, 1, 1]),
      positionIds: new Int32Array([0, 1, 0, 1])
    }
    const actual = T.zeros(4, 5)
    const actualWeights = T.zeros(4, 4)

    const expected = referenceCausalAttention(q, k, v, layout)

    Ops.fusedScaledDotProductAttentionIntoSync(q, k, v, actual, {
      causalMask: true,
      layout,
      weightsOut: actualWeights
    })

    expectClose(actual, expected.attended)
    expectClose(actualWeights, expected.weights)
  })

  test("fused SDPA backward matches composed causal path", () => {
    const q = makePatternedTensor(4, 6, 15)
    const k = makePatternedTensor(4, 6, 16)
    const v = makePatternedTensor(4, 5, 17)
    const dOut = makePatternedTensor(4, 5, 18)
    const workspace = new TensorWorkspace()
    const actualDQ = T.zeros(4, 6)
    const actualDK = T.zeros(4, 6)
    const actualDV = T.zeros(4, 5)

    const expected = referenceSdpaBackward(q, k, v, dOut)

    Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, actualDQ, actualDK, actualDV, {
      causalMask: true,
      workspace
    })

    expectClose(actualDQ, expected.dQ)
    expectClose(actualDK, expected.dK)
    expectClose(actualDV, expected.dV)
  })

  test("fused SDPA backward respects sequence layout masks", () => {
    const q = makePatternedTensor(4, 5, 19)
    const k = makePatternedTensor(4, 5, 20)
    const v = makePatternedTensor(4, 4, 21)
    const dOut = makePatternedTensor(4, 4, 22)
    const layout = {
      totalTokens: 4,
      sequenceIds: new Int32Array([0, 0, 1, 1]),
      positionIds: new Int32Array([0, 1, 0, 1])
    }
    const actualDQ = T.fromArray(4, 5, new Float32Array(20).fill(7))
    const actualDK = T.fromArray(4, 5, new Float32Array(20).fill(7))
    const actualDV = T.fromArray(4, 4, new Float32Array(16).fill(7))

    const expected = referenceSdpaBackward(q, k, v, dOut, layout)

    Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, actualDQ, actualDK, actualDV, {
      causalMask: true,
      layout
    })

    expectClose(actualDQ, expected.dQ)
    expectClose(actualDK, expected.dK)
    expectClose(actualDV, expected.dV)
  })

  test("fused SDPA backward rejects aliased outputs", () => {
    const q = makePatternedTensor(2, 3, 23)
    const k = makePatternedTensor(2, 3, 24)
    const v = makePatternedTensor(2, 2, 25)
    const dOut = makePatternedTensor(2, 2, 26)
    const dK = T.zeros(2, 3)
    const dV = T.zeros(2, 2)

    expect(() =>
      Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, q, dK, dV, {
        causalMask: true
      })
    ).toThrow("must not alias input storage")
  })

  test("fused SDPA backward rejects overlapping output subarrays from shared storage", () => {
    const q = makePatternedTensor(2, 3, 27)
    const k = makePatternedTensor(2, 3, 28)
    const v = makePatternedTensor(2, 2, 29)
    const dOut = makePatternedTensor(2, 2, 30)
    const shared = new Float32Array(16)
    const dQ = T.make(2, 3, shared.subarray(0, 6))
    const dK = T.make(2, 3, shared.subarray(4, 10))
    const dV = T.make(2, 2, shared.subarray(10, 14))

    expect(() =>
      Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, dQ, dK, dV, {
        causalMask: true
      })
    ).toThrow("must not alias each other")
  })

  test("fused SDPA backward allows disjoint output subarrays from shared storage", () => {
    const q = makePatternedTensor(2, 3, 31)
    const k = makePatternedTensor(2, 3, 32)
    const v = makePatternedTensor(2, 2, 33)
    const dOut = makePatternedTensor(2, 2, 34)
    const shared = new Float32Array(16)
    const actualDQ = T.make(2, 3, shared.subarray(0, 6))
    const actualDK = T.make(2, 3, shared.subarray(6, 12))
    const actualDV = T.make(2, 2, shared.subarray(12, 16))
    const expected = referenceSdpaBackward(q, k, v, dOut)

    Ops.fusedSdpaBackwardIntoSync(q, k, v, dOut, actualDQ, actualDK, actualDV, {
      causalMask: true
    })

    expectClose(actualDQ, expected.dQ)
    expectClose(actualDK, expected.dK)
    expectClose(actualDV, expected.dV)
  })

  test("gatherRowsInto matches allocating gatherRows", () => {
    const embeddings = T.fromArray(4, 2, [1, 10, 2, 20, 3, 30, 4, 40])
    const tokenIds = [3, 1, 0]

    const expected = runEffect(Ops.gatherRows(embeddings, tokenIds))
    const actual = T.zeros(3, 2)
    runEffect(Ops.gatherRowsInto(embeddings, tokenIds, actual))

    expectClose(actual, expected)
  })

  test("gatherRowsInto rejects out-of-range row ids", () => {
    const embeddings = T.fromArray(4, 2, [1, 10, 2, 20, 3, 30, 4, 40])
    const actual = T.zeros(1, 2)

    const error = runEffectFail(Ops.gatherRowsInto(embeddings, [4], actual))

    expect(error.message).toContain("tokenId 4 out of bounds")
  })

  test("crossEntropyLossAndDLogitsFromLogits matches softmax-based path", () => {
    const logits = T.fromArray(3, 4, [1, 3, 2, 0, -1, 0, 2, 4, 5, 1, 0, -2])
    const targets = [1, 3, 0]

    const probs = Ops.softmaxRows(logits)
    const expected = crossEntropyLossAndDLogits(probs, targets)
    const actual = crossEntropyLossAndDLogitsFromLogits(logits, targets)

    expect(actual.loss).toBeCloseTo(expected.loss, 6)
    expectClose(actual.grads, expected.grads)
  })
})
