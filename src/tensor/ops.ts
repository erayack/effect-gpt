import * as Effect from "effect/Effect"
import type { Tensor2D } from "./Tensor2D"
import * as T from "./Tensor2D"
import type { Rng } from "./random"
import type { RandomServiceId } from "../services/Random"
import { Random } from "../services/Random"

export class ShapeError extends Error {
  readonly _tag = "ShapeError"
  constructor(message: string) {
    super(message)
    this.name = "ShapeError"
  }
}

export const matMul = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> => {
  if (a.cols !== b.rows) {
    return Effect.fail(new ShapeError(`matMul: a.cols (${a.cols}) !== b.rows (${b.rows})`))
  }

  return Effect.sync(() => {
    const aRows = a.rows
    const aCols = a.cols
    const bCols = b.cols

    const aData = a.data
    const bData = b.data
    const resultData = new Float32Array(aRows * bCols)

    for (let i = 0; i < aRows; i++) {
      const resultRowOffset = i * bCols
      const aRowOffset = i * aCols

      for (let k = 0; k < aCols; k++) {
        const aVal = aData[aRowOffset + k]
        const bRowOffset = k * bCols

        let j = 0
        const limit = bCols - (bCols % 4)
        for (; j < limit; j += 4) {
          resultData[resultRowOffset + j] += aVal * bData[bRowOffset + j]
          resultData[resultRowOffset + j + 1] += aVal * bData[bRowOffset + j + 1]
          resultData[resultRowOffset + j + 2] += aVal * bData[bRowOffset + j + 2]
          resultData[resultRowOffset + j + 3] += aVal * bData[bRowOffset + j + 3]
        }
        for (; j < bCols; j++) {
          resultData[resultRowOffset + j] += aVal * bData[bRowOffset + j]
        }
      }
    }

    return T.make(aRows, bCols, resultData)
  })
}

export const add = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new ShapeError(`add: shapes (${a.rows},${a.cols}) and (${b.rows},${b.cols}) do not match`)
    }
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i] + b.data[i]
    }
    return T.make(a.rows, a.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const sub = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new ShapeError(`sub: shapes (${a.rows},${a.cols}) and (${b.rows},${b.cols}) do not match`)
    }
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i] - b.data[i]
    }
    return T.make(a.rows, a.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const mul = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new ShapeError(`mul: shapes (${a.rows},${a.cols}) and (${b.rows},${b.cols}) do not match`)
    }
    const data = new Float32Array(a.data.length)
    for (let i = 0; i < data.length; i++) {
      data[i] = a.data[i] * b.data[i]
    }
    return T.make(a.rows, a.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const div = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new ShapeError(`div: shapes (${a.rows},${a.cols}) and (${b.rows},${b.cols}) do not match`)
    }
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

export const mulScalar = (t: Tensor2D, scalar: number): Tensor2D => {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    data[i] = t.data[i] * scalar
  }
  return T.make(t.rows, t.cols, data)
}

export const addRowBias = (matrix: Tensor2D, bias: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (bias.rows !== 1 || bias.cols !== matrix.cols) {
      throw new ShapeError(`addRowBias: bias shape (${bias.rows},${bias.cols}) incompatible with matrix cols ${matrix.cols}`)
    }
    const rows = matrix.rows
    const cols = matrix.cols
    const matrixData = matrix.data
    const biasData = bias.data
    const data = new Float32Array(matrix.data.length)
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols
      for (let j = 0; j < cols; j++) {
        data[rowOffset + j] = matrixData[rowOffset + j] + biasData[j]
      }
    }
    return T.make(rows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

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
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = new Float32Array(cols)
  for (let j = 0; j < cols; j++) {
    let sum = 0
    for (let i = 0; i < rows; i++) {
      sum += tData[i * cols + j]
    }
    data[j] = sum
  }
  return T.make(1, cols, data)
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
  const data = new Float32Array(t.data.length)
  const tData = t.data
  const cols = t.cols

  for (let i = 0; i < t.rows; i++) {
    const rowOffset = i * cols
    let maxVal = -Infinity

    for (let j = 0; j < cols; j++) {
      const val = tData[rowOffset + j]
      if (val > maxVal) maxVal = val
    }

    let sumExp = 0
    for (let j = 0; j < cols; j++) {
      const exp = Math.exp(tData[rowOffset + j] - maxVal)
      data[rowOffset + j] = exp
      sumExp += exp
    }

    for (let j = 0; j < cols; j++) {
      data[rowOffset + j] /= sumExp
    }
  }

  return T.make(t.rows, t.cols, data)
}

export const transpose = (t: Tensor2D): Tensor2D => {
  const rows = t.rows
  const cols = t.cols
  const tData = t.data
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols
    for (let j = 0; j < cols; j++) {
      data[j * rows + i] = tData[rowOffset + j]
    }
  }
  return T.make(cols, rows, data)
}

export const gatherRows = (embeddings: Tensor2D, tokenIds: ArrayLike<number>): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    const cols = embeddings.cols
    const rows = embeddings.rows
    const embeddingsData = embeddings.data
    const data = new Float32Array(tokenIds.length * cols)
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
    return T.make(tokenIds.length, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const sliceRows = (t: Tensor2D, start: number, end: number): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (start < 0 || end > t.rows || start >= end) {
      throw new ShapeError(`sliceRows: invalid range [${start}, ${end}) for tensor with ${t.rows} rows`)
    }
    const cols = t.cols
    const tData = t.data
    const numRows = end - start
    const data = new Float32Array(numRows * cols)
    for (let i = 0; i < numRows; i++) {
      const sourceOffset = (start + i) * cols
      const targetOffset = i * cols
      for (let j = 0; j < cols; j++) {
        data[targetOffset + j] = tData[sourceOffset + j]
      }
    }
    return T.make(numRows, cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const relu = (t: Tensor2D): Tensor2D => {
  const data = new Float32Array(t.data.length)
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.max(0, t.data[i])
  }
  return T.make(t.rows, t.cols, data)
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
