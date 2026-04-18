import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import * as Ops from "../tensor/ops"

export const softmaxRows = (logits: Tensor2D): Tensor2D => Ops.softmaxRows(logits)

export const crossEntropyLoss = (probs: Tensor2D, targetIds: ReadonlyArray<number>): number => {
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

export const dLogits = (probs: Tensor2D, targetIds: ReadonlyArray<number>): Tensor2D => {
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
  targetIds: ReadonlyArray<number>
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
