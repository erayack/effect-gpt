import type { Tensor2D } from "../tensor/Tensor2D"
import * as T from "../tensor/Tensor2D"
import { ShapeError } from "../tensor/ops"

export class Adam {
  readonly beta1 = 0.9
  readonly beta2 = 0.999
  readonly epsilon = 1e-8
  timestep = 0
  m: Tensor2D
  v: Tensor2D

  private constructor(rows: number, cols: number) {
    this.m = T.zeros(rows, cols)
    this.v = T.zeros(rows, cols)
  }

  static make(rows: number, cols: number): Adam {
    return new Adam(rows, cols)
  }

  step(params: Tensor2D, grads: Tensor2D, lr: number): void {
    if (params.rows !== grads.rows || params.cols !== grads.cols) {
      throw new ShapeError(
        `Adam.step: params shape (${params.rows},${params.cols}) != grads shape (${grads.rows},${grads.cols})`
      )
    }
    if (this.m.rows !== params.rows || this.m.cols !== params.cols) {
      throw new ShapeError(
        `Adam.step: optimizer shape (${this.m.rows},${this.m.cols}) != params shape (${params.rows},${params.cols})`
      )
    }

    this.timestep += 1
    const beta1 = this.beta1
    const beta2 = this.beta2
    const oneMinusB1 = 1 - beta1
    const oneMinusB2 = 1 - beta2

    const mData = this.m.data
    const vData = this.v.data
    const pData = params.data
    const gData = grads.data

    for (let i = 0; i < gData.length; i++) {
      const g = gData[i]
      mData[i] = mData[i] * beta1 + g * oneMinusB1
      vData[i] = vData[i] * beta2 + g * g * oneMinusB2
    }

    const mHatScale = 1 - Math.pow(beta1, this.timestep)
    const vHatScale = 1 - Math.pow(beta2, this.timestep)

    for (let i = 0; i < pData.length; i++) {
      const mHat = mData[i] / mHatScale
      const vHat = vData[i] / vHatScale
      const update = mHat / (Math.sqrt(vHat) + this.epsilon)
      pData[i] -= lr * update
    }
  }
}
