import type { Tensor2D } from "./Tensor2D"
import * as T from "./Tensor2D"

export class TensorWorkspace {
  private readonly tensors = new Map<string, Float32Array>()
  private readonly vectors = new Map<string, Float32Array>()

  borrowTensor(name: string, rows: number, cols: number): Tensor2D {
    const length = rows * cols
    const existing = this.tensors.get(name)
    if (existing && existing.length >= length) {
      return T.make(rows, cols, existing.subarray(0, length))
    }

    const next = new Float32Array(length)
    this.tensors.set(name, next)
    return T.make(rows, cols, next)
  }

  borrowVector(name: string, length: number): Float32Array {
    const existing = this.vectors.get(name)
    if (existing && existing.length === length) {
      return existing
    }

    const next = new Float32Array(length)
    this.vectors.set(name, next)
    return next
  }

  borrowVectorAtLeast(name: string, minLength: number): Float32Array {
    const existing = this.vectors.get(name)
    if (existing && existing.length >= minLength) {
      return existing
    }

    const next = new Float32Array(minLength)
    this.vectors.set(name, next)
    return next
  }
}
