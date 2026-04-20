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
  for (let j = 0; j < cols; j++) {
    let sum = 0
    for (let i = 0; i < rows; i++) {
      sum += tData[i * cols + j]
    }
    data[j] = sum
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
    const sequenceIds = layout.sequenceIds
    const positionIds = layout.positionIds
    const scoresData = scores.data
    for (let i = 0; i < seqLen; i++) {
      const rowOffset = i * seqLen
      const querySequenceId = sequenceIds[i]
      const queryPositionId = positionIds[i]
      for (let j = 0; j < seqLen; j++) {
        if (sequenceIds[j] !== querySequenceId || positionIds[j] > queryPositionId) {
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
