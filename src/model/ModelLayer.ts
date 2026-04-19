import type * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import type { ShapeError } from "../tensor/ops"

export interface SequenceLayout {
  readonly totalTokens: number
  readonly sequenceLengths: ReadonlyArray<number>
  readonly sequenceIds: Int32Array
  readonly positionIds: Int32Array
}

export interface LayerForwardContext {
  readonly sequenceLayout?: SequenceLayout
}

export interface ModelLayer {
  readonly _tag: string
  readonly parametersCount: number
  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError>
  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError>
}
