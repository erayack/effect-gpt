import type { Tensor2D } from "../tensor/Tensor2D"

export const clipGlobalL2 = (grads: Tensor2D, maxNorm: number): void => {
  let sumSq = 0
  for (let i = 0; i < grads.data.length; i++) {
    const v = grads.data[i]
    sumSq += v * v
  }

  const maxNormSq = maxNorm * maxNorm
  if (sumSq > maxNormSq) {
    const scale = maxNorm / Math.sqrt(sumSq)
    for (let i = 0; i < grads.data.length; i++) {
      grads.data[i] *= scale
    }
  }
}
