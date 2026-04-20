import type { Tensor2D } from "./Tensor2D"
import type { TensorWorkspace } from "./Workspace"

export interface MatMulRequest {
  readonly a: Tensor2D
  readonly b: Tensor2D
  readonly out: Tensor2D
  readonly transposeA: boolean
  readonly transposeB: boolean
  readonly workspace?: TensorWorkspace
}

const SMALL_PRODUCT_THRESHOLD = 32_768
const BLOCK_K = 64
const BLOCK_N = 64

const getPanelBuffer = (workspace: TensorWorkspace | undefined): Float32Array =>
  workspace?.borrowVectorAtLeast("gemm.rhsPanel", BLOCK_K * BLOCK_N) ?? new Float32Array(BLOCK_K * BLOCK_N)

const multiplyIntoSimple = ({ a, b, out, transposeA, transposeB }: MatMulRequest): void => {
  const outRows = out.rows
  const outCols = out.cols
  const sharedDim = transposeA ? a.rows : a.cols
  const outData = out.data
  const aData = a.data
  const bData = b.data

  outData.fill(0)

  if (!transposeA && !transposeB) {
    for (let row = 0; row < outRows; row++) {
      const outRowOffset = row * outCols
      const aRowOffset = row * a.cols

      for (let k = 0; k < sharedDim; k++) {
        const aValue = aData[aRowOffset + k]!
        const bRowOffset = k * b.cols
        let col = 0
        const limit = outCols - (outCols % 4)
        for (; col < limit; col += 4) {
          outData[outRowOffset + col] += aValue * bData[bRowOffset + col]!
          outData[outRowOffset + col + 1] += aValue * bData[bRowOffset + col + 1]!
          outData[outRowOffset + col + 2] += aValue * bData[bRowOffset + col + 2]!
          outData[outRowOffset + col + 3] += aValue * bData[bRowOffset + col + 3]!
        }
        for (; col < outCols; col++) {
          outData[outRowOffset + col] += aValue * bData[bRowOffset + col]!
        }
      }
    }
    return
  }

  if (!transposeA && transposeB) {
    for (let row = 0; row < outRows; row++) {
      const outRowOffset = row * outCols
      const aRowOffset = row * a.cols
      for (let col = 0; col < outCols; col++) {
        const bRowOffset = col * b.cols
        let sum = 0
        for (let k = 0; k < sharedDim; k++) {
          sum += aData[aRowOffset + k]! * bData[bRowOffset + k]!
        }
        outData[outRowOffset + col] = sum
      }
    }
    return
  }

  if (transposeA && !transposeB) {
    for (let row = 0; row < outRows; row++) {
      const outRowOffset = row * outCols
      for (let col = 0; col < outCols; col++) {
        let sum = 0
        for (let k = 0; k < sharedDim; k++) {
          sum += aData[k * a.cols + row]! * bData[k * b.cols + col]!
        }
        outData[outRowOffset + col] = sum
      }
    }
    return
  }

  for (let row = 0; row < outRows; row++) {
    const outRowOffset = row * outCols
    for (let col = 0; col < outCols; col++) {
      const bRowOffset = col * b.cols
      let sum = 0
      for (let k = 0; k < sharedDim; k++) {
        sum += aData[k * a.cols + row]! * bData[bRowOffset + k]!
      }
      outData[outRowOffset + col] = sum
    }
  }
}

const packBPanel = (
  b: Tensor2D,
  transposeB: boolean,
  kStart: number,
  colStart: number,
  kBlock: number,
  nBlock: number,
  panel: Float32Array
): void => {
  const bData = b.data

  if (!transposeB) {
    for (let k = 0; k < kBlock; k++) {
      const sourceOffset = (kStart + k) * b.cols + colStart
      const targetOffset = k * nBlock
      for (let col = 0; col < nBlock; col++) {
        panel[targetOffset + col] = bData[sourceOffset + col]!
      }
    }
    return
  }

  for (let k = 0; k < kBlock; k++) {
    const sourceCol = kStart + k
    const targetOffset = k * nBlock
    for (let col = 0; col < nBlock; col++) {
      panel[targetOffset + col] = bData[(colStart + col) * b.cols + sourceCol]!
    }
  }
}

const multiplyIntoBlocked = ({ a, b, out, transposeA, transposeB, workspace }: MatMulRequest): void => {
  const outRows = out.rows
  const outCols = out.cols
  const sharedDim = transposeA ? a.rows : a.cols
  const outData = out.data
  const aData = a.data

  outData.fill(0)
  const panel = getPanelBuffer(workspace)

  for (let colStart = 0; colStart < outCols; colStart += BLOCK_N) {
    const nBlock = Math.min(BLOCK_N, outCols - colStart)

    for (let kStart = 0; kStart < sharedDim; kStart += BLOCK_K) {
      const kBlock = Math.min(BLOCK_K, sharedDim - kStart)
      packBPanel(b, transposeB, kStart, colStart, kBlock, nBlock, panel)

      for (let row = 0; row < outRows; row++) {
        const outRowOffset = row * outCols + colStart

        if (!transposeA) {
          const aRowOffset = row * a.cols + kStart
          for (let k = 0; k < kBlock; k++) {
            const aValue = aData[aRowOffset + k]!
            const panelOffset = k * nBlock
            let col = 0
            const limit = nBlock - (nBlock % 4)
            for (; col < limit; col += 4) {
              outData[outRowOffset + col] += aValue * panel[panelOffset + col]!
              outData[outRowOffset + col + 1] += aValue * panel[panelOffset + col + 1]!
              outData[outRowOffset + col + 2] += aValue * panel[panelOffset + col + 2]!
              outData[outRowOffset + col + 3] += aValue * panel[panelOffset + col + 3]!
            }
            for (; col < nBlock; col++) {
              outData[outRowOffset + col] += aValue * panel[panelOffset + col]!
            }
          }
          continue
        }

        for (let k = 0; k < kBlock; k++) {
          const aValue = aData[(kStart + k) * a.cols + row]!
          const panelOffset = k * nBlock
          let col = 0
          const limit = nBlock - (nBlock % 4)
          for (; col < limit; col += 4) {
            outData[outRowOffset + col] += aValue * panel[panelOffset + col]!
            outData[outRowOffset + col + 1] += aValue * panel[panelOffset + col + 1]!
            outData[outRowOffset + col + 2] += aValue * panel[panelOffset + col + 2]!
            outData[outRowOffset + col + 3] += aValue * panel[panelOffset + col + 3]!
          }
          for (; col < nBlock; col++) {
            outData[outRowOffset + col] += aValue * panel[panelOffset + col]!
          }
        }
      }
    }
  }
}

export const gemmMultiplyInto = (request: MatMulRequest): void => {
  const outRows = request.out.rows
  const outCols = request.out.cols
  const sharedDim = request.transposeA ? request.a.rows : request.a.cols
  const work = outRows * outCols * sharedDim

  if (work < SMALL_PRODUCT_THRESHOLD) {
    multiplyIntoSimple(request)
    return
  }

  multiplyIntoBlocked(request)
}
