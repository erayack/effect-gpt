/**
 * Custom tensor assertion helpers for deterministic testing.
 */
import { expect } from "bun:test"
import type { Tensor2D } from "../../../src/tensor/Tensor2D"

/** Default epsilon for floating-point comparisons. */
export const DEFAULT_EPSILON = 1e-5

/**
 * Asserts that a tensor has the expected shape.
 */
export const expectShape = (tensor: Tensor2D, shape: [number, number]): void => {
  expect(tensor.rows).toBe(shape[0])
  expect(tensor.cols).toBe(shape[1])
}

/**
 * Asserts that two tensors have the same shape and all elements are within epsilon.
 */
export const expectClose = (
  actual: Tensor2D,
  expected: Tensor2D,
  epsilon: number = DEFAULT_EPSILON
): void => {
  expectShape(actual, [expected.rows, expected.cols])

  for (let i = 0; i < actual.data.length; i++) {
    const a = actual.data[i]
    const e = expected.data[i]
    const diff = Math.abs(a - e)
    if (diff > epsilon) {
      const row = Math.floor(i / actual.cols)
      const col = i % actual.cols
      throw new Error(
        `Tensor mismatch at [${row}, ${col}]: got ${a}, expected ${e}, diff ${diff} > epsilon ${epsilon}`
      )
    }
  }
}

/**
 * Asserts that all elements of a tensor are within epsilon of a target value.
 */
export const expectAllClose = (
  tensor: Tensor2D,
  value: number,
  epsilon: number = DEFAULT_EPSILON
): void => {
  for (let i = 0; i < tensor.data.length; i++) {
    const v = tensor.data[i]
    const diff = Math.abs(v - value)
    if (diff > epsilon) {
      const row = Math.floor(i / tensor.cols)
      const col = i % tensor.cols
      throw new Error(
        `Tensor element at [${row}, ${col}]: got ${v}, expected ~${value}, diff ${diff} > epsilon ${epsilon}`
      )
    }
  }
}

/**
 * Asserts that two tensors are NOT equal (at least one element differs by more than epsilon).
 */
export const expectNotClose = (
  actual: Tensor2D,
  expected: Tensor2D,
  epsilon: number = DEFAULT_EPSILON
): void => {
  if (actual.rows !== expected.rows || actual.cols !== expected.cols) {
    return // Different shapes means not equal
  }

  for (let i = 0; i < actual.data.length; i++) {
    const diff = Math.abs(actual.data[i] - expected.data[i])
    if (diff > epsilon) {
      return // Found a difference
    }
  }

  throw new Error("Expected tensors to differ, but they are equal within epsilon")
}

/**
 * Asserts that a tensor contains finite values (no NaN or Infinity).
 */
export const expectFinite = (tensor: Tensor2D): void => {
  for (let i = 0; i < tensor.data.length; i++) {
    const v = tensor.data[i]
    if (!Number.isFinite(v)) {
      const row = Math.floor(i / tensor.cols)
      const col = i % tensor.cols
      throw new Error(`Non-finite value at [${row}, ${col}]: ${v}`)
    }
  }
}
