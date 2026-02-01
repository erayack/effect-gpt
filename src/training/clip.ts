import type { Tensor2D } from "../tensor/Tensor2D"

export const clipGlobalL2 = (grads: Tensor2D, maxNorm: number): void => {
  let sumSq = 0
  for (let i = 0; i < grads.data.length; i++) {
    const v = grads.data[i]
    sumSq += v * v
  }
  const norm = Math.sqrt(sumSq)
  if (norm > maxNorm) {
    const scale = maxNorm / norm
    for (let i = 0; i < grads.data.length; i++) {
      grads.data[i] *= scale
    }
  }
}
