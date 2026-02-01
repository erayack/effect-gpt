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

export const matMul = (a: Tensor2D, b: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (a.cols !== b.rows) {
      throw new ShapeError(`matMul: a.cols (${a.cols}) !== b.rows (${b.rows})`)
    }
    const result = T.zeros(a.rows, b.cols)
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < b.cols; j++) {
        let sum = 0
        for (let k = 0; k < a.cols; k++) {
          sum += T.get(a, i, k) * T.get(b, k, j)
        }
        T.set(result, i, j, sum)
      }
    }
    return result
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

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
    const data = new Float32Array(matrix.data.length)
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        data[i * matrix.cols + j] = T.get(matrix, i, j) + bias.data[j]
      }
    }
    return T.make(matrix.rows, matrix.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const meanRows = (t: Tensor2D): Tensor2D => {
  const data = new Float32Array(t.rows)
  for (let i = 0; i < t.rows; i++) {
    let sum = 0
    for (let j = 0; j < t.cols; j++) {
      sum += T.get(t, i, j)
    }
    data[i] = sum / t.cols
  }
  return T.make(t.rows, 1, data)
}

export const sumCols = (t: Tensor2D): Tensor2D => {
  const data = new Float32Array(t.cols)
  for (let j = 0; j < t.cols; j++) {
    let sum = 0
    for (let i = 0; i < t.rows; i++) {
      sum += T.get(t, i, j)
    }
    data[j] = sum
  }
  return T.make(1, t.cols, data)
}

export const meanCols = (t: Tensor2D): Tensor2D => {
  const data = new Float32Array(t.cols)
  const scale = 1 / t.rows
  for (let j = 0; j < t.cols; j++) {
    let sum = 0
    for (let i = 0; i < t.rows; i++) {
      sum += T.get(t, i, j)
    }
    data[j] = sum * scale
  }
  return T.make(1, t.cols, data)
}

export const stdRows = (t: Tensor2D): Tensor2D => {
  const means = meanRows(t)
  const data = new Float32Array(t.rows)
  for (let i = 0; i < t.rows; i++) {
    const mean = means.data[i]
    let sumSq = 0
    for (let j = 0; j < t.cols; j++) {
      const diff = T.get(t, i, j) - mean
      sumSq += diff * diff
    }
    data[i] = Math.sqrt(sumSq / t.cols)
  }
  return T.make(t.rows, 1, data)
}

export const varRows = (t: Tensor2D): Tensor2D => {
  const means = meanRows(t)
  const data = new Float32Array(t.rows)
  for (let i = 0; i < t.rows; i++) {
    const mean = means.data[i]
    let sumSq = 0
    for (let j = 0; j < t.cols; j++) {
      const diff = T.get(t, i, j) - mean
      sumSq += diff * diff
    }
    data[i] = sumSq / t.cols
  }
  return T.make(t.rows, 1, data)
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
  for (let i = 0; i < t.rows; i++) {
    let maxVal = -Infinity
    for (let j = 0; j < t.cols; j++) {
      const val = T.get(t, i, j)
      if (val > maxVal) maxVal = val
    }
    let sumExp = 0
    for (let j = 0; j < t.cols; j++) {
      const exp = Math.exp(T.get(t, i, j) - maxVal)
      data[i * t.cols + j] = exp
      sumExp += exp
    }
    for (let j = 0; j < t.cols; j++) {
      data[i * t.cols + j] /= sumExp
    }
  }
  return T.make(t.rows, t.cols, data)
}

export const transpose = (t: Tensor2D): Tensor2D => {
  const data = new Float32Array(t.rows * t.cols)
  for (let i = 0; i < t.rows; i++) {
    for (let j = 0; j < t.cols; j++) {
      data[j * t.rows + i] = T.get(t, i, j)
    }
  }
  return T.make(t.cols, t.rows, data)
}

export const gatherRows = (embeddings: Tensor2D, tokenIds: ReadonlyArray<number>): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    const data = new Float32Array(tokenIds.length * embeddings.cols)
    for (let i = 0; i < tokenIds.length; i++) {
      const tokenId = tokenIds[i]
      if (tokenId < 0 || tokenId >= embeddings.rows) {
        throw new ShapeError(`gatherRows: tokenId ${tokenId} out of bounds [0, ${embeddings.rows})`)
      }
      for (let j = 0; j < embeddings.cols; j++) {
        data[i * embeddings.cols + j] = T.get(embeddings, tokenId, j)
      }
    }
    return T.make(tokenIds.length, embeddings.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const sliceRows = (t: Tensor2D, start: number, end: number): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (start < 0 || end > t.rows || start >= end) {
      throw new ShapeError(`sliceRows: invalid range [${start}, ${end}) for tensor with ${t.rows} rows`)
    }
    const numRows = end - start
    const data = new Float32Array(numRows * t.cols)
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, start + i, j)
      }
    }
    return T.make(numRows, t.cols, data)
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
  for (let i = 0; i < t.rows; i++) {
    let maxIdx = 0
    let maxVal = T.get(t, i, 0)
    for (let j = 1; j < t.cols; j++) {
      const val = T.get(t, i, j)
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
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, i, j) - col.data[i]
      }
    }
    return T.make(t.rows, t.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastDivCol = (t: Tensor2D, col: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (col.cols !== 1 || col.rows !== t.rows) {
      throw new ShapeError(`broadcastDivCol: col shape (${col.rows},${col.cols}) incompatible with tensor rows ${t.rows}`)
    }
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, i, j) / col.data[i]
      }
    }
    return T.make(t.rows, t.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastMulCol = (t: Tensor2D, col: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (col.cols !== 1 || col.rows !== t.rows) {
      throw new ShapeError(`broadcastMulCol: col shape (${col.rows},${col.cols}) incompatible with tensor rows ${t.rows}`)
    }
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, i, j) * col.data[i]
      }
    }
    return T.make(t.rows, t.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastMulRow = (t: Tensor2D, row: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (row.rows !== 1 || row.cols !== t.cols) {
      throw new ShapeError(`broadcastMulRow: row shape (${row.rows},${row.cols}) incompatible with tensor cols ${t.cols}`)
    }
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, i, j) * row.data[j]
      }
    }
    return T.make(t.rows, t.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const broadcastAddRow = (t: Tensor2D, row: Tensor2D): Effect.Effect<Tensor2D, ShapeError> =>
  Effect.sync(() => {
    if (row.rows !== 1 || row.cols !== t.cols) {
      throw new ShapeError(`broadcastAddRow: row shape (${row.rows},${row.cols}) incompatible with tensor cols ${t.cols}`)
    }
    const data = new Float32Array(t.data.length)
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) {
        data[i * t.cols + j] = T.get(t, i, j) + row.data[j]
      }
    }
    return T.make(t.rows, t.cols, data)
  }).pipe(Effect.catchAllDefect((e) => Effect.fail(e as ShapeError)))

export const initNormal = (rows: number, cols: number, mean: number, std: number, rng?: Rng): Tensor2D => {
  const rand = rng ? () => rng.next() : Math.random
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
