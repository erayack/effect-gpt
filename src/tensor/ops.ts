import * as Effect from "effect/Effect"
import type { Tensor2D } from "./Tensor2D"
import * as T from "./Tensor2D"
import type { Rng } from "./random"
import type { RandomServiceId } from "../services/Random"
import { Random } from "../services/Random"
import { gemmMultiplyInto } from "./gemm"
import type { TensorWorkspace } from "./Workspace"

export class ShapeError extends Error {
  readonly _tag = "ShapeError"
  constructor(message: string) {
    super(message)
    this.name = "ShapeError"
  }
}

const validateSameShape = (op: string, a: Tensor2D, b: Tensor2D): void => {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new ShapeError(`${op}: shapes (${a.rows},${a.cols}) and (${b.rows},${b.cols}) do not match`)
  }
}

const validateOutputShape = (op: string, out: Tensor2D, rows: number, cols: number): void => {
  if (out.rows !== rows || out.cols !== cols) {
    throw new ShapeError(`${op}: output shape (${out.rows},${out.cols}) does not match expected (${rows},${cols})`)
  }
}

export interface MatMulOptions {
  readonly transposeA?: boolean
  readonly transposeB?: boolean
  // Optional scratch workspace for callers that want to reuse temporary buffers across repeated multiplies.
  readonly workspace?: TensorWorkspace
}

export interface FusedScaledDotProductAttentionOptions {
  readonly causalMask?: boolean
  readonly layout?: {
    readonly sequenceIds: Int32Array
    readonly positionIds: Int32Array
    readonly totalTokens: number
  }
  readonly weightsOut?: Tensor2D
  readonly workspace?: TensorWorkspace
}

export interface FusedScaledDotProductAttentionBackwardOptions {
  readonly causalMask?: boolean
  readonly layout?: {
    readonly sequenceIds: Int32Array
    readonly positionIds: Int32Array
    readonly totalTokens: number
  }
  readonly workspace?: TensorWorkspace
}

export const toShapeError = (error: unknown): ShapeError =>
  error instanceof ShapeError ? error : (error as ShapeError)

export const syncShapeEffect = <A>(thunk: () => A): Effect.Effect<A, ShapeError> =>
  Effect.try({
    try: thunk,
    catch: toShapeError
  })

const resolveMatMulShape = (
  a: Tensor2D,
  b: Tensor2D,
  options?: MatMulOptions
): {
  readonly transposeA: boolean
  readonly transposeB: boolean
  readonly outRows: number
  readonly sharedDim: number
  readonly bRows: number
  readonly outCols: number
} => {
  const transposeA = options?.transposeA === true
  const transposeB = options?.transposeB === true
  const outRows = transposeA ? a.cols : a.rows
  const sharedDim = transposeA ? a.rows : a.cols
  const bRows = transposeB ? b.cols : b.rows
  const outCols = transposeB ? b.rows : b.cols
  return { transposeA, transposeB, outRows, sharedDim, bRows, outCols }
}

export const matMulIntoSync = (
  a: Tensor2D,
  b: Tensor2D,
  out: Tensor2D,
  options?: MatMulOptions
): void => {
  const { transposeA, transposeB, outRows, sharedDim, bRows, outCols } = resolveMatMulShape(a, b, options)

  if (sharedDim !== bRows) {
    throw new ShapeError(
      `matMul: effective inner dimensions do not match (${outRows},${sharedDim}) x (${bRows},${outCols})`
    )
  }
  if (out.rows !== outRows || out.cols !== outCols) {
    throw new ShapeError(`matMul: output shape (${out.rows},${out.cols}) does not match expected (${outRows},${outCols})`)
  }
  if (out.data === a.data || out.data === b.data) {
    throw new ShapeError("matMul: output tensor must not alias input storage")
  }

  const request = {
    a,
    b,
    out,
    transposeA,
    transposeB,
    ...(options?.workspace ? { workspace: options.workspace } : {})
  }
  gemmMultiplyInto(request)
}

export const matMulInto = (
  a: Tensor2D,
  b: Tensor2D,
  out: Tensor2D,
  options?: MatMulOptions
): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    matMulIntoSync(a, b, out, options)
  })

export const matMulSync = (a: Tensor2D, b: Tensor2D, options?: MatMulOptions): Tensor2D => {
  const { outRows, outCols } = resolveMatMulShape(a, b, options)
  const out = T.zeros(outRows, outCols)
  matMulIntoSync(a, b, out, options)
  return out
}

export const matMul = (a: Tensor2D, b: Tensor2D, options?: MatMulOptions): Effect.Effect<Tensor2D, ShapeError> => {
  return syncShapeEffect(() => matMulSync(a, b, options))
}

export const addIntoSync = (a: Tensor2D, b: Tensor2D, out: Tensor2D): void => {
  validateSameShape("add", a, b)
  validateOutputShape("add", out, a.rows, a.cols)
  const data = out.data
  for (let i = 0; i < data.length; i++) {
    data[i] = a.data[i] + b.data[i]
  }
}

export const addInto = (a: Tensor2D, b: Tensor2D, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    addIntoSync(a, b, out)
  })

export const addSync = (a: Tensor2D, b: Tensor2D): Tensor2D => {
  const out = T.zeros(a.rows, a.cols)
  addIntoSync(a, b, out)
  return out
}

export const add = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  syncShapeEffect(() => addSync(a, b))

export const sub = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    validateSameShape("sub", a, b)
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i] - b.data[i]
    }
    return T.make(a.rows, a.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const mulIntoSync = (a: Tensor2D, b: Tensor2D, out: Tensor2D): void => {
  validateSameShape("mul", a, b)
  validateOutputShape("mul", out, a.rows, a.cols)
  const data = out.data
  for (let i = 0; i < data.length; i++) {
    data[i] = a.data[i] * b.data[i]
  }
}

export const mulInto = (a: Tensor2D, b: Tensor2D, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    mulIntoSync(a, b, out)
  })

export const mul = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.gen(function* () {
    const out = T.zeros(a.rows, a.cols)
    yield* mulInto(a, b, out)
    return out
  })

export const div = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    validateSameShape("div", a, b)
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i] / b.data[i]
    }
    return T.make(a.rows, a.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const addScalar = (t: Tensor2D, scalar: number): Tensor2D => {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    data[i] = t.data[i] + scalar
  }
  return T.make(t.rows, t.cols, data)
}

export const mulScalarInPlace = (t: Tensor2D, scalar: number): void => {
  for (let i = 0; i < t.data.length; i++) {
    t.data[i] *= scalar
  }
}

export const mulScalar = (t: Tensor2D, scalar: number): Tensor2D => {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    data[i] = t.data[i] * scalar
  }
  return T.make(t.rows, t.cols, data)
}

export const addRowBiasInPlaceSync = (matrix: Tensor2D, bias: Tensor2D): void => {
  if (bias.rows !== 1 || bias.cols !== matrix.cols) {
    throw new ShapeError(`addRowBias: bias shape (${bias.rows},${bias.cols}) incompatible with matrix cols ${matrix.cols}`)
  }
  const rows = matrix.rows
  const cols = matrix.cols
  const matrixData = matrix.data
  const biasData = bias.data
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    for (let j = 0; j < cols; j++) {
      matrixData[rowOffset + j] += biasData[j]
    }
  }
}

export const addRowBiasInPlace = (matrix: Tensor2D, bias: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    addRowBiasInPlaceSync(matrix, bias)
  })

export const addRowBiasIntoSync = (matrix: Tensor2D, bias: Tensor2D, out: Tensor2D): void => {
  if (bias.rows !== 1 || bias.cols !== matrix.cols) {
    throw new ShapeError(`addRowBias: bias shape (${bias.rows},${bias.cols}) incompatible with matrix cols ${matrix.cols}`)
  }
  validateOutputShape("addRowBias", out, matrix.rows, matrix.cols)
  const rows = matrix.rows
  const cols = matrix.cols
  const matrixData = matrix.data
  const biasData = bias.data
  const outData = out.data
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    for (let j = 0; j < cols; j++) {
      outData[rowOffset + j] = matrixData[rowOffset + j] + biasData[j]
    }
  }
}

export const addRowBiasInto = (matrix: Tensor2D, bias: Tensor2D, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    addRowBiasIntoSync(matrix, bias, out)
  })

export const addRowBias = (matrix: Tensor2D, bias: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  syncShapeEffect(() => {
    const out = T.zeros(matrix.rows, matrix.cols)
    addRowBiasIntoSync(matrix, bias, out)
    return out
  })

export const meanRows = (t: Tensor2D): Tensor2D => {
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = new Float32Array(rows)
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    let sum = 0
    for (let j = 0; j < cols; j++) {
      sum += tData[rowOffset + j]
    }
    data[i] = sum / cols
  }
  return T.make(rows, 1, data)
}

export const sumCols = (t: Tensor2D): Tensor2D => {
  const out = T.zeros(1, t.cols)
  sumColsInto(t, out)
  return out
}

export const sumColsInto = (t: Tensor2D, out: Tensor2D): void => {
  validateOutputShape("sumCols", out, 1, t.cols)
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = out.data
  data.fill(0)

  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    let j = 0
    const limit = cols - (cols % 4)
    for (; j < limit; j += 4) {
      data[j] += tData[rowOffset + j]!
      data[j + 1] += tData[rowOffset + j + 1]!
      data[j + 2] += tData[rowOffset + j + 2]!
      data[j + 3] += tData[rowOffset + j + 3]!
    }
    for (; j < cols; j++) {
      data[j] += tData[rowOffset + j]!
    }
  }
}

export const meanCols = (t: Tensor2D): Tensor2D => {
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = new Float32Array(cols)
  const scale = 1 / rows
  for (let j = 0; j < cols; j++) {
    let sum = 0
    for (let i = 0; i < rows; i++) {
      sum += tData[i * cols + j]
    }
    data[j] = sum * scale
  }
  return T.make(1, cols, data)
}

export const stdRows = (t: Tensor2D): Tensor2D => {
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const means = meanRows(t)
  const data = new Float32Array(rows)
  for (let i = 0; i < rows; i++) {
    const mean = means.data[i]
    const rowOffset = i * cols
    let sumSq = 0
    for (let j = 0; j < cols; j++) {
      const diff = tData[rowOffset + j] - mean
      sumSq += diff * diff
    }
    data[i] = Math.sqrt(sumSq / cols)
  }
  return T.make(rows, 1, data)
}

export const varRows = (t: Tensor2D): Tensor2D => {
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const means = meanRows(t)
  const data = new Float32Array(rows)
  for (let i = 0; i < rows; i++) {
    const mean = means.data[i]
    const rowOffset = i * cols
    let sumSq = 0
    for (let j = 0; j < cols; j++) {
      const diff = tData[rowOffset + j] - mean
      sumSq += diff * diff
    }
    data[i] = sumSq / cols
  }
  return T.make(rows, 1, data)
}

export const mapScalar = (t: Tensor2D, fn: (val: number) => number): Tensor2D => {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    data[i] = fn(t.data[i])
  }
  return T.make(t.rows, t.cols, data)
}

export const softmaxRows = (t: Tensor2D): Tensor2D => {
  const out = T.zeros(t.rows, t.cols)
  softmaxRowsInto(t, out)
  return out
}

export const softmaxRowsInto = (input: Tensor2D, out: Tensor2D): void => {
  validateOutputShape("softmaxRows", out, input.rows, input.cols)
  const data = out.data
  const inputData = input.data
  const cols = input.cols

  for (let i = 0; i < input.rows; i++) {
    const rowOffset = i * cols
    let maxVal = -Infinity

    for (let j = 0; j < cols; j++) {
      const val = inputData[rowOffset + j]
      if (val > maxVal) maxVal = val
    }

    let sumExp = 0
    for (let j = 0; j < cols; j++) {
      const exp = Math.exp(inputData[rowOffset + j] - maxVal)
      data[rowOffset + j] = exp
      sumExp += exp
    }

    for (let j = 0; j < cols; j++) {
      data[rowOffset + j] /= sumExp
    }
  }
}

export const softmaxRowsInPlace = (t: Tensor2D): void => {
  const data = t.data
  const cols = t.cols

  for (let i = 0; i < t.rows; i++) {
    const rowOffset = i * cols
    let maxVal = -Infinity

    for (let j = 0; j < cols; j++) {
      const val = data[rowOffset + j]
      if (val > maxVal) maxVal = val
    }

    let sumExp = 0
    for (let j = 0; j < cols; j++) {
      const exp = Math.exp(data[rowOffset + j] - maxVal)
      data[rowOffset + j] = exp
      sumExp += exp
    }

    for (let j = 0; j < cols; j++) {
      data[rowOffset + j] /= sumExp
    }
  }
}

const isMaskedAttentionPosition = (
  queryIndex: number,
  keyIndex: number,
  causalMask: boolean,
  layout?: {
    readonly sequenceIds: Int32Array
    readonly positionIds: Int32Array
    readonly totalTokens: number
  }
): boolean => {
  if (!causalMask) {
    return false
  }
  if (!layout) {
    return keyIndex > queryIndex
  }
  return (
    layout.sequenceIds[keyIndex] !== layout.sequenceIds[queryIndex] ||
    layout.positionIds[keyIndex] > layout.positionIds[queryIndex]
  )
}

const validateNoAttentionAliasing = (
  op: string,
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  out: Tensor2D,
  weightsOut?: Tensor2D
): void => {
  if (out.data === q.data || out.data === k.data || out.data === v.data) {
    throw new ShapeError(`${op}: output tensor must not alias input storage`)
  }
  if (!weightsOut) {
    return
  }
  if (
    weightsOut.data === q.data ||
    weightsOut.data === k.data ||
    weightsOut.data === v.data ||
    weightsOut.data === out.data
  ) {
    throw new ShapeError(`${op}: weights tensor must not alias input or output storage`)
  }
}

const typedArrayRangesOverlap = (a: Float32Array, b: Float32Array): boolean => {
  if (a.buffer !== b.buffer) {
    return false
  }

  const aStart = a.byteOffset
  const aEnd = aStart + a.byteLength
  const bStart = b.byteOffset
  const bEnd = bStart + b.byteLength
  return aStart < bEnd && bStart < aEnd
}

const validateNoAttentionBackwardAliasing = (
  op: string,
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  dOut: Tensor2D,
  dQ: Tensor2D,
  dK: Tensor2D,
  dV: Tensor2D
): void => {
  if (
    typedArrayRangesOverlap(dQ.data, q.data) ||
    typedArrayRangesOverlap(dQ.data, k.data) ||
    typedArrayRangesOverlap(dQ.data, v.data) ||
    typedArrayRangesOverlap(dQ.data, dOut.data) ||
    typedArrayRangesOverlap(dK.data, q.data) ||
    typedArrayRangesOverlap(dK.data, k.data) ||
    typedArrayRangesOverlap(dK.data, v.data) ||
    typedArrayRangesOverlap(dK.data, dOut.data) ||
    typedArrayRangesOverlap(dV.data, q.data) ||
    typedArrayRangesOverlap(dV.data, k.data) ||
    typedArrayRangesOverlap(dV.data, v.data) ||
    typedArrayRangesOverlap(dV.data, dOut.data)
  ) {
    throw new ShapeError(`${op}: output tensors must not alias input storage`)
  }
  if (
    typedArrayRangesOverlap(dQ.data, dK.data) ||
    typedArrayRangesOverlap(dQ.data, dV.data) ||
    typedArrayRangesOverlap(dK.data, dV.data)
  ) {
    throw new ShapeError(`${op}: output tensors must not alias each other`)
  }
}

const zeroAttentionBackwardOutputs = (dQ: Float32Array, dK: Float32Array, dV: Float32Array): void => {
  if (dQ.buffer === dK.buffer && dQ.buffer === dV.buffer) {
    const start = Math.min(dQ.byteOffset, dK.byteOffset, dV.byteOffset)
    const end = Math.max(
      dQ.byteOffset + dQ.byteLength,
      dK.byteOffset + dK.byteLength,
      dV.byteOffset + dV.byteLength
    )
    new Float32Array(dQ.buffer, start, (end - start) / Float32Array.BYTES_PER_ELEMENT).fill(0)
    return
  }

  dQ.fill(0)
  dK.fill(0)
  dV.fill(0)
}

export const fusedScaledDotProductAttentionIntoSync = (
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  out: Tensor2D,
  options?: FusedScaledDotProductAttentionOptions
): void => {
  if (q.cols !== k.cols) {
    throw new ShapeError(`fusedScaledDotProductAttention: q cols ${q.cols} do not match k cols ${k.cols}`)
  }
  if (k.rows !== v.rows) {
    throw new ShapeError(`fusedScaledDotProductAttention: k rows ${k.rows} do not match v rows ${v.rows}`)
  }
  if (out.rows !== q.rows || out.cols !== v.cols) {
    throw new ShapeError(
      `fusedScaledDotProductAttention: output shape (${out.rows},${out.cols}) does not match expected (${q.rows},${v.cols})`
    )
  }

  const weightsOut = options?.weightsOut
  if (weightsOut && (weightsOut.rows !== q.rows || weightsOut.cols !== k.rows)) {
    throw new ShapeError(
      `fusedScaledDotProductAttention: weights shape (${weightsOut.rows},${weightsOut.cols}) does not match expected (${q.rows},${k.rows})`
    )
  }
  validateNoAttentionAliasing("fusedScaledDotProductAttention", q, k, v, out, weightsOut)

  const causalMask = options?.causalMask === true
  const layout = options?.layout
  if (layout && !causalMask) {
    throw new ShapeError("fusedScaledDotProductAttention: layout requires causalMask=true")
  }
  if (layout && (q.rows !== k.rows || layout.totalTokens !== q.rows)) {
    throw new ShapeError(
      `fusedScaledDotProductAttention: layout totalTokens (${layout.totalTokens}) incompatible with q/k shapes ${q.rows}x${k.rows}`
    )
  }

  const scratch =
    options?.workspace?.borrowVectorAtLeast("fusedAttentionScores", k.rows) ??
    new Float32Array(k.rows)

  const qCols = q.cols
  const vCols = v.cols
  const scale = 1 / Math.sqrt(qCols)
  const qData = q.data
  const kData = k.data
  const vData = v.data
  const outData = out.data
  outData.fill(0)

  for (let queryRow = 0; queryRow < q.rows; queryRow++) {
    const qOffset = queryRow * qCols
    let maxScore = -Infinity

    for (let keyRow = 0; keyRow < k.rows; keyRow++) {
      if (isMaskedAttentionPosition(queryRow, keyRow, causalMask, layout)) {
        scratch[keyRow] = -Infinity
        continue
      }

      const kOffset = keyRow * qCols
      let score = 0
      for (let col = 0; col < qCols; col++) {
        score += qData[qOffset + col]! * kData[kOffset + col]!
      }
      score *= scale
      scratch[keyRow] = score
      if (score > maxScore) {
        maxScore = score
      }
    }

    let sumExp = 0
    for (let keyRow = 0; keyRow < k.rows; keyRow++) {
      const score = scratch[keyRow]!
      if (score === -Infinity) {
        scratch[keyRow] = 0
        if (weightsOut) {
          weightsOut.data[queryRow * k.rows + keyRow] = 0
        }
        continue
      }
      const weight = Math.exp(score - maxScore)
      scratch[keyRow] = weight
      sumExp += weight
    }

    const outOffset = queryRow * vCols
    for (let keyRow = 0; keyRow < k.rows; keyRow++) {
      const weight = sumExp === 0 ? 0 : scratch[keyRow]! / sumExp
      if (weightsOut) {
        weightsOut.data[queryRow * k.rows + keyRow] = weight
      }
      if (weight === 0) {
        continue
      }
      const vOffset = keyRow * vCols
      for (let col = 0; col < vCols; col++) {
        outData[outOffset + col] += weight * vData[vOffset + col]!
      }
    }
  }
}

export const fusedScaledDotProductAttentionSync = (
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  options?: FusedScaledDotProductAttentionOptions
): Tensor2D => {
  const out = T.zeros(q.rows, v.cols)
  fusedScaledDotProductAttentionIntoSync(q, k, v, out, options)
  return out
}

export const fusedSdpaBackwardIntoSync = (
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  dOut: Tensor2D,
  dQ: Tensor2D,
  dK: Tensor2D,
  dV: Tensor2D,
  options?: FusedScaledDotProductAttentionBackwardOptions
): void => {
  if (q.cols !== k.cols) {
    throw new ShapeError(`fusedSdpaBackward: q cols ${q.cols} do not match k cols ${k.cols}`)
  }
  if (k.rows !== v.rows) {
    throw new ShapeError(`fusedSdpaBackward: k rows ${k.rows} do not match v rows ${v.rows}`)
  }
  if (dOut.rows !== q.rows || dOut.cols !== v.cols) {
    throw new ShapeError(
      `fusedSdpaBackward: dOut shape (${dOut.rows},${dOut.cols}) does not match expected (${q.rows},${v.cols})`
    )
  }
  validateOutputShape("fusedSdpaBackward", dQ, q.rows, q.cols)
  validateOutputShape("fusedSdpaBackward", dK, k.rows, k.cols)
  validateOutputShape("fusedSdpaBackward", dV, v.rows, v.cols)
  validateNoAttentionBackwardAliasing("fusedSdpaBackward", q, k, v, dOut, dQ, dK, dV)

  const causalMask = options?.causalMask === true
  const layout = options?.layout
  if (layout && !causalMask) {
    throw new ShapeError("fusedSdpaBackward: layout requires causalMask=true")
  }
  if (layout && (q.rows !== k.rows || layout.totalTokens !== q.rows)) {
    throw new ShapeError(`fusedSdpaBackward: layout totalTokens (${layout.totalTokens}) incompatible with q/k shapes ${q.rows}x${k.rows}`)
  }

  const weightsScratch =
    options?.workspace?.borrowVectorAtLeast("fusedSdpaBackwardWeights", k.rows) ??
    new Float32Array(k.rows)
  const gradScratch =
    options?.workspace?.borrowVectorAtLeast("fusedSdpaBackwardGradScores", k.rows) ??
    new Float32Array(k.rows)

  const qRows = q.rows
  const qCols = q.cols
  const kRows = k.rows
  const vCols = v.cols
  const scale = 1 / Math.sqrt(qCols)
  const qData = q.data
  const kData = k.data
  const vData = v.data
  const dOutData = dOut.data
  const dQData = dQ.data
  const dKData = dK.data
  const dVData = dV.data

  zeroAttentionBackwardOutputs(dQData, dKData, dVData)

  for (let queryRow = 0; queryRow < qRows; queryRow++) {
    const qOffset = queryRow * qCols
    const dOutOffset = queryRow * vCols
    let maxScore = -Infinity

    for (let keyRow = 0; keyRow < kRows; keyRow++) {
      if (isMaskedAttentionPosition(queryRow, keyRow, causalMask, layout)) {
        weightsScratch[keyRow] = -Infinity
        continue
      }

      const kOffset = keyRow * qCols
      let score = 0
      for (let col = 0; col < qCols; col++) {
        score += qData[qOffset + col]! * kData[kOffset + col]!
      }
      score *= scale
      weightsScratch[keyRow] = score
      if (score > maxScore) {
        maxScore = score
      }
    }

    let sumExp = 0
    for (let keyRow = 0; keyRow < kRows; keyRow++) {
      const score = weightsScratch[keyRow]!
      if (score === -Infinity) {
        weightsScratch[keyRow] = 0
        continue
      }
      const weight = Math.exp(score - maxScore)
      weightsScratch[keyRow] = weight
      sumExp += weight
    }

    let softmaxDot = 0
    for (let keyRow = 0; keyRow < kRows; keyRow++) {
      const weight = sumExp === 0 ? 0 : weightsScratch[keyRow]! / sumExp
      weightsScratch[keyRow] = weight
      if (weight === 0) {
        gradScratch[keyRow] = 0
        continue
      }

      const vOffset = keyRow * vCols
      let gradWeight = 0
      for (let col = 0; col < vCols; col++) {
        const dOutValue = dOutData[dOutOffset + col]!
        gradWeight += dOutValue * vData[vOffset + col]!
        dVData[vOffset + col] += weight * dOutValue
      }
      gradScratch[keyRow] = gradWeight
      softmaxDot += weight * gradWeight
    }

    for (let keyRow = 0; keyRow < kRows; keyRow++) {
      const weight = weightsScratch[keyRow]!
      if (weight === 0) {
        gradScratch[keyRow] = 0
        continue
      }
      gradScratch[keyRow] = weight * (gradScratch[keyRow]! - softmaxDot) * scale
    }

    for (let keyRow = 0; keyRow < kRows; keyRow++) {
      const dScore = gradScratch[keyRow]!
      if (dScore === 0) {
        continue
      }
      const kOffset = keyRow * qCols
      for (let col = 0; col < qCols; col++) {
        dQData[qOffset + col] += dScore * kData[kOffset + col]!
        dKData[kOffset + col] += dScore * qData[qOffset + col]!
      }
    }
  }
}

export const fusedSdpaBackwardInto = (
  q: Tensor2D,
  k: Tensor2D,
  v: Tensor2D,
  dOut: Tensor2D,
  dQ: Tensor2D,
  dK: Tensor2D,
  dV: Tensor2D,
  options?: FusedScaledDotProductAttentionBackwardOptions
): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    fusedSdpaBackwardIntoSync(q, k, v, dOut, dQ, dK, dV, options)
  })

export const transpose = (t: Tensor2D): Tensor2D => {
  const out = T.zeros(t.cols, t.rows)
  transposeInto(t, out)
  return out
}

export const transposeInto = (t: Tensor2D, out: Tensor2D): void => {
  validateOutputShape("transpose", out, t.cols, t.rows)
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = out.data
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    for (let j = 0; j < cols; j++) {
      data[j * rows + i] = tData[rowOffset + j]
    }
  }
}

export const gatherRowsIntoSync = (embeddings: Tensor2D, tokenIds: ArrayLike<number>, out: Tensor2D): void => {
  validateOutputShape("gatherRows", out, tokenIds.length, embeddings.cols)
  const cols = embeddings.cols
  const rows = embeddings.rows
  const embeddingsData = embeddings.data
  const data = out.data
  for (let i = 0; i < tokenIds.length; i++) {
    const tokenId = tokenIds[i]
    if (tokenId < 0 || tokenId >= rows) {
      throw new ShapeError(`gatherRows: tokenId ${tokenId} out of bounds [0, ${rows})`)
    }
    const sourceOffset = tokenId * cols
    const targetOffset = i * cols
    for (let j = 0; j < cols; j++) {
      data[targetOffset + j] = embeddingsData[sourceOffset + j]
    }
  }
}

export const gatherRowsInto = (embeddings: Tensor2D, tokenIds: ArrayLike<number>, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    gatherRowsIntoSync(embeddings, tokenIds, out)
  })

export const gatherRows = (embeddings: Tensor2D, tokenIds: ArrayLike<number>): Effect.Effect<Tensor2D, ShapeError> =>
  syncShapeEffect(() => {
    const out = T.zeros(tokenIds.length, embeddings.cols)
    gatherRowsIntoSync(embeddings, tokenIds, out)
    return out
  })

export const sliceRowsIntoSync = (t: Tensor2D, start: number, end: number, out: Tensor2D): void => {
  if (start < 0 || end > t.rows || start >= end) {
    throw new ShapeError(`sliceRows: invalid range [${start}, ${end}) for tensor with ${t.rows} rows`)
  }
  const cols = t.cols
  const tData = t.data
  const numRows = end - start
  validateOutputShape("sliceRows", out, numRows, cols)
  const data = out.data
  for (let i = 0; i < numRows; i++) {
    const sourceOffset = (start + i) * cols
    const targetOffset = i * cols
    for (let j = 0; j < cols; j++) {
      data[targetOffset + j] = tData[sourceOffset + j]
    }
  }
}

export const sliceRowsInto = (t: Tensor2D, start: number, end: number, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    sliceRowsIntoSync(t, start, end, out)
  })

export const sliceRows = (t: Tensor2D, start: number, end: number): Effect.Effect<Tensor2D, ShapeError> =>
  syncShapeEffect(() => {
    if (start < 0 || end > t.rows || start >= end) {
      throw new ShapeError(`sliceRows: invalid range [${start}, ${end}) for tensor with ${t.rows} rows`)
    }
    const out = T.zeros(end - start, t.cols)
    sliceRowsIntoSync(t, start, end, out)
    return out
  })

export const rowAsMatrixIntoSync = (t: Tensor2D, row: number, out: Tensor2D): void => {
  if (row < 0 || row >= t.rows) {
    throw new ShapeError(`rowAsMatrix: row ${row} out of bounds for tensor with ${t.rows} rows`)
  }
  validateOutputShape("rowAsMatrix", out, 1, t.cols)
  const start = row * t.cols
  out.data.set(t.data.subarray(start, start + t.cols))
}

export const rowAsMatrixInto = (t: Tensor2D, row: number, out: Tensor2D): Effect.Effect<void, ShapeError> =>
  syncShapeEffect(() => {
    rowAsMatrixIntoSync(t, row, out)
  })

export const rowAsMatrixSync = (t: Tensor2D, row: number): Tensor2D => {
  const out = T.zeros(1, t.cols)
  rowAsMatrixIntoSync(t, row, out)
  return out
}

export const rowAsMatrix = (t: Tensor2D, row: number): Effect.Effect<Tensor2D, ShapeError> =>
  syncShapeEffect(() => rowAsMatrixSync(t, row))

export const reluInPlace = (t: Tensor2D): void => {
  for (let i = 0; i < t.data.length; i++) {
    t.data[i] = Math.max(0, t.data[i])
  }
}

export const reluInto = (input: Tensor2D, out: Tensor2D): void => {
  validateOutputShape("relu", out, input.rows, input.cols)
  for (let i = 0; i < out.data.length; i++) {
    out.data[i] = Math.max(0, input.data[i])
  }
}

export const relu = (t: Tensor2D): Tensor2D => {
  const out = T.zeros(t.rows, t.cols)
  reluInto(t, out)
  return out
}

export const argmaxRows = (t: Tensor2D): ReadonlyArray<number> => {
  const result: Array<number> = []
  const tData = t.data
  const cols = t.cols

  for (let i = 0; i < t.rows; i++) {
    const rowOffset = i * cols
    let maxIdx = 0
    let maxVal = tData[rowOffset]

    for (let j = 1; j < cols; j++) {
      const val = tData[rowOffset + j]
      if (val > maxVal) {
        maxVal = val
        maxIdx = j
      }
    }

    result.push(maxIdx)
  }

  return result
}

export const broadcastSubCol = (t: Tensor2D, col: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (col.cols !== 1 || col.rows !== t.rows) {
      throw new ShapeError(`broadcastSubCol: col shape (${col.rows},${col.cols}) incompatible with tensor rows ${t.rows}`)
    }
    const rows = t.rows
    const cols = t.cols
    const tData = t.data
    const colData = col.data
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      const colValue = colData[i]
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = tData[rowOffset + j] - colValue
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const maskCausalInPlace = (scores: Tensor2D, layout?: { sequenceIds: Int32Array; positionIds: Int32Array; totalTokens: number }): void => {
  const seqLen = scores.rows
  if (scores.cols !== seqLen) {
    throw new ShapeError(`maskCausalInPlace: expected square scores matrix, received ${scores.rows}x${scores.cols}`)
  }

  if (layout) {
    if (layout.totalTokens !== seqLen) {
      throw new ShapeError(
        `maskCausalInPlace: layout totalTokens (${layout.totalTokens}) incompatible with scores shape ${seqLen}x${scores.cols}`
      )
    }
    const scoresData = scores.data
    for (let i = 0; i < seqLen; i++) {
      const rowOffset = i * seqLen
      for (let j = 0; j < seqLen; j++) {
        if (isMaskedAttentionPosition(i, j, true, layout)) {
          scoresData[rowOffset + j] = -Infinity
        }
      }
    }
    return
  }

  for (let i = 0; i < seqLen; i++) {
    const rowOffset = i * seqLen
    for (let j = i + 1; j < seqLen; j++) {
      scores.data[rowOffset + j] = -Infinity
    }
  }
}

export const broadcastDivCol = (t: Tensor2D, col: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (col.cols !== 1 || col.rows !== t.rows) {
      throw new ShapeError(`broadcastDivCol: col shape (${col.rows},${col.cols}) incompatible with tensor rows ${t.rows}`)
    }
    const rows = t.rows
    const cols = t.cols
    const tData = t.data
    const colData = col.data
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      const colValue = colData[i]
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = tData[rowOffset + j] / colValue
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastMulCol = (t: Tensor2D, col: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (col.cols !== 1 || col.rows !== t.rows) {
      throw new ShapeError(`broadcastMulCol: col shape (${col.rows},${col.cols}) incompatible with tensor rows ${t.rows}`)
    }
    const rows = t.rows
    const cols = t.cols
    const tData = t.data
    const colData = col.data
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      const colValue = colData[i]
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = tData[rowOffset + j] * colValue
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastMulRow = (t: Tensor2D, row: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (row.rows !== 1 || row.cols !== t.cols) {
      throw new ShapeError(`broadcastMulRow: row shape (${row.rows},${row.cols}) incompatible with tensor cols ${t.cols}`)
    }
    const rows = t.rows
    const cols = t.cols
    const tData = t.data
    const rowData = row.data
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = tData[rowOffset + j] * rowData[j]
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastAddRow = (t: Tensor2D, row: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (row.rows !== 1 || row.cols !== t.cols) {
      throw new ShapeError(`broadcastAddRow: row shape (${row.rows},${row.cols}) incompatible with tensor cols ${t.cols}`)
    }
    const rows = t.rows
    const cols = t.cols
    const tData = t.data
    const rowData = row.data
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = tData[rowOffset + j] + rowData[j]
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const initNormal = (rows: number, cols: number, mean: number, std: number, rng: Rng): Tensor2D => {
  const rand = () => rng.next()
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < data.length; i++) {
    let u1 = rand()
    let u2 = rand()
    while (u1 === 0) u1 = rand()
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    data[i] = mean + std * z
  }
  return T.make(rows, cols, data)
}

export const initNormalEffect = (
  rows: number,
  cols: number,
  mean: number,
  std: number
): Effect.Effect<Tensor2D, never, RandomServiceId> =>
  Effect.flatMap(Random, (random) =>
    Effect.gen(function* () {
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < data.length; i++) {
        const value = yield* random.nextGaussian(mean, std)
        data[i] = value
      }
      return T.make(rows, cols, data)
    })
  )
