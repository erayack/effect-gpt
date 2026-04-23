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

  private validateParams(params: Tensor2D): void {
    if (this.m.rows !== params.rows || this.m.cols !== params.cols) {
      throw new ShapeError(
        `Adam.step: optimizer shape (${this.m.rows},${this.m.cols}) != params shape (${params.rows},${params.cols})`
      )
    }
  }

  private nextScales(): {
    oneMinusB1: number
    oneMinusB2: number
    invMHatScale: number
    invVHatScale: number
  } {
    this.timestep += 1
    const beta1 = this.beta1
    const beta2 = this.beta2
    return {
      oneMinusB1: 1 - beta1,
      oneMinusB2: 1 - beta2,
      invMHatScale: 1 / (1 - Math.pow(beta1, this.timestep)),
      invVHatScale: 1 / (1 - Math.pow(beta2, this.timestep))
    }
  }

  step(params: Tensor2D, grads: Tensor2D, lr: number): void {
    if (params.rows !== grads.rows || params.cols !== grads.cols) {
      throw new ShapeError(
        `Adam.step: params shape (${params.rows},${params.cols}) != grads shape (${grads.rows},${grads.cols})`
      )
    }
    this.validateParams(params)

    const { oneMinusB1, oneMinusB2, invMHatScale, invVHatScale } = this.nextScales()
    const beta1 = this.beta1
    const beta2 = this.beta2
    const epsilon = this.epsilon

    const mData = this.m.data
    const vData = this.v.data
    const pData = params.data
    const gData = grads.data

    const length = gData.length
    for (let i = 0; i < length; i++) {
      const g = gData[i]
      const m = mData[i] * beta1 + g * oneMinusB1
      const v = vData[i] * beta2 + g * g * oneMinusB2
      mData[i] = m
      vData[i] = v
      const mHat = m * invMHatScale
      const vHat = v * invVHatScale
      pData[i] -= lr * (mHat / (Math.sqrt(vHat) + epsilon))
    }
  }

  stepRows(params: Tensor2D, rowIndices: ReadonlyArray<number>, gradRows: Float32Array, lr: number): void {
    if (params.cols === 0) {
      return
    }
    if (gradRows.length !== rowIndices.length * params.cols) {
      throw new ShapeError(
        `Adam.stepRows: gradRows length ${gradRows.length} != rowIndices.length * params.cols (${rowIndices.length * params.cols})`
      )
    }
    this.validateParams(params)

    const { oneMinusB1, oneMinusB2, invMHatScale, invVHatScale } = this.nextScales()
    const beta1 = this.beta1
    const beta2 = this.beta2
    const epsilon = this.epsilon
    const cols = params.cols
    const rowCount = rowIndices.length

    const mData = this.m.data
    const vData = this.v.data
    const pData = params.data

    for (let row = 0; row < rowCount; row++) {
      const paramRow = rowIndices[row]!
      let index = paramRow * cols
      const end = index + cols
      let gradIndex = row * cols
      for (; index < end; index++, gradIndex++) {
        const g = gradRows[gradIndex]
        const m = mData[index] * beta1 + g * oneMinusB1
        const v = vData[index] * beta2 + g * g * oneMinusB2
        mData[index] = m
        vData[index] = v
        const mHat = m * invMHatScale
        const vHat = v * invVHatScale
        pData[index] -= lr * (mHat / (Math.sqrt(vHat) + epsilon))
      }
    }
  }
}
