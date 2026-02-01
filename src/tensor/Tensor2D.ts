export interface Tensor2D {
  readonly rows: number
  readonly cols: number
  readonly data: Float32Array
}

export const make = (rows: number, cols: number, data: Float32Array): Tensor2D => {
  if (data.length !== rows * cols) {
    throw new Error(`Data length ${data.length} does not match shape ${rows}x${cols}`)
  }
  return { rows, cols, data }
}

export const zeros = (rows: number, cols: number): Tensor2D => {
  return { rows, cols, data: new Float32Array(rows * cols) }
}

export const ones = (rows: number, cols: number): Tensor2D => {
  const data = new Float32Array(rows * cols)
  data.fill(1)
  return { rows, cols, data }
}

export const clone = (t: Tensor2D): Tensor2D => {
  return { rows: t.rows, cols: t.cols, data: new Float32Array(t.data) }
}

export const get = (t: Tensor2D, row: number, col: number): number => {
  return t.data[row * t.cols + col]
}

export const set = (t: Tensor2D, row: number, col: number, value: number): void => {
  t.data[row * t.cols + col] = value
}

export const fromArray = (rows: number, cols: number, arr: ArrayLike<number>): Tensor2D => {
  const data = new Float32Array(arr)
  if (data.length !== rows * cols) {
    throw new Error(`Array length ${data.length} does not match shape ${rows}x${cols}`)
  }
  return { rows, cols, data }
}
