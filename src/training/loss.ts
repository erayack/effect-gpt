import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"

export const softmaxRows = (logits: Tensor2D): Tensor2D => Ops.softmaxRows(logits)

export const crossEntropyLoss = (probs: Tensor2D, targetIds: ArrayLike<number>): number => {
  if (probs.rows !== targetIds.length) {
    throw new Ops.ShapeError(`crossEntropyLoss: probs.rows (${probs.rows}) !== targetIds.length (${targetIds.length})`)
  }
  let loss = 0
  for (let i = 0; i < probs.rows; i++) {
    const idx = targetIds[i]
    const prob = probs.data[i * probs.cols + idx]
    const clamped = prob < 1e-15 ? 1e-15 : prob
    loss -= Math.log(clamped)
  }
  return loss / targetIds.length
}

export const dLogits = (probs: Tensor2D, targetIds: ArrayLike<number>): Tensor2D => {
  if (probs.rows !== targetIds.length) {
    throw new Ops.ShapeError(`dLogits: probs.rows (${probs.rows}) !== targetIds.length (${targetIds.length})`)
  }
  const data = new Float32Array(probs.data)
  for (let i = 0; i < probs.rows; i++) {
    const idx = targetIds[i]
    data[i * probs.cols + idx] -= 1
  }
  const scale = 1 / targetIds.length
  for (let i = 0; i < data.length; i++) {
    data[i] *= scale
  }
  return T.make(probs.rows, probs.cols, data)
}

export const crossEntropyLossAndDLogits = (
  probs: Tensor2D,
  targetIds: ArrayLike<number>
): { loss: number; grads: Tensor2D } => {
  if (probs.rows !== targetIds.length) {
    throw new Ops.ShapeError(
      `crossEntropyLossAndDLogits: probs.rows (${probs.rows}) !== targetIds.length (${targetIds.length})`
    )
  }

  const data = new Float32Array(probs.data)
  let loss = 0
  for (let i = 0; i < probs.rows; i++) {
    const idx = targetIds[i]
    const offset = i * probs.cols + idx
    const prob = probs.data[offset]
    const clamped = prob < 1e-15 ? 1e-15 : prob
    loss -= Math.log(clamped)
    data[offset] -= 1
  }

  const scale = 1 / targetIds.length
  for (let i = 0; i < data.length; i++) {
    data[i] *= scale
  }

  return {
    loss: loss / targetIds.length,
    grads: T.make(probs.rows, probs.cols, data)
  }
}

export const crossEntropyLossAndDLogitsFromLogitsInto = (
  logits: Tensor2D,
  targetIds: ArrayLike<number>,
  gradsOut: Tensor2D
): number => {
  if (logits.rows !== targetIds.length) {
    throw new Ops.ShapeError(
      `crossEntropyLossAndDLogitsFromLogitsInto: logits.rows (${logits.rows}) !== targetIds.length (${targetIds.length})`
    )
  }
  if (gradsOut.rows !== logits.rows || gradsOut.cols !== logits.cols) {
    throw new Ops.ShapeError(
      `crossEntropyLossAndDLogitsFromLogitsInto: gradsOut shape (${gradsOut.rows},${gradsOut.cols}) !== logits shape (${logits.rows},${logits.cols})`
    )
  }

  const data = gradsOut.data
  const cols = logits.cols
  let loss = 0

  for (let row = 0; row < logits.rows; row++) {
    const rowOffset = row * cols
    let maxVal = -Infinity
    for (let col = 0; col < cols; col++) {
      const value = logits.data[rowOffset + col]!
      if (value > maxVal) {
        maxVal = value
      }
    }

    let sumExp = 0
    for (let col = 0; col < cols; col++) {
      const exp = Math.exp(logits.data[rowOffset + col]! - maxVal)
      data[rowOffset + col] = exp
      sumExp += exp
    }

    for (let col = 0; col < cols; col++) {
      data[rowOffset + col] /= sumExp
    }

    const targetIndex = targetIds[row]!
    const targetOffset = rowOffset + targetIndex
    const prob = data[targetOffset]!
    const clamped = prob < 1e-15 ? 1e-15 : prob
    loss -= Math.log(clamped)
    data[targetOffset] -= 1
  }

  const scale = 1 / targetIds.length
  for (let i = 0; i < data.length; i++) {
    data[i] *= scale
  }

  return loss / targetIds.length
}

export const crossEntropyLossAndDLogitsFromLogits = (
  logits: Tensor2D,
  targetIds: ArrayLike<number>
): { loss: number; grads: Tensor2D } => {
  const grads = T.zeros(logits.rows, logits.cols)
  const loss = crossEntropyLossAndDLogitsFromLogitsInto(logits, targetIds, grads)
  return { loss, grads }
}
