import * as Effect from "effect/Effect"
import type { Tensor2D } from "../tensor/Tensor2D"
import type { ShapeError } from "../tensor/ops"

export type LayerCacheKey = number | string | symbol

export interface SequenceLayout {
  readonly totalTokens: number
  readonly sequenceLengths: ReadonlyArray<number>
  readonly sequenceIds: Int32Array
  readonly positionIds: Int32Array
}

export interface LayerForwardContext {
  readonly sequenceLayout?: SequenceLayout
  readonly cacheKey?: LayerCacheKey
  readonly captureCache?: boolean
}

export interface ModelLayer {
  readonly _tag: string
  readonly parametersCount: number
  forward(input: Tensor2D, context?: LayerForwardContext): Effect.Effect<Tensor2D, ShapeError>
  backward(dOut: Tensor2D, lr: number): Effect.Effect<Tensor2D, ShapeError>
}

export interface SyncModelLayer extends ModelLayer {
  forwardSync(input: Tensor2D, context?: LayerForwardContext): Tensor2D
  backwardSync(dOut: Tensor2D, lr: number, cacheKey?: LayerCacheKey): Tensor2D
}

const hasSyncModelLayer = (layer: ModelLayer): layer is SyncModelLayer =>
  "forwardSync" in layer &&
  typeof (layer as { forwardSync?: unknown }).forwardSync === "function" &&
  "backwardSync" in layer &&
  typeof (layer as { backwardSync?: unknown }).backwardSync === "function"

const hasSyncNetwork = (layers: ReadonlyArray<ModelLayer>): layers is ReadonlyArray<SyncModelLayer> =>
  layers.every(hasSyncModelLayer)

export const runForwardPass = (
  layers: ReadonlyArray<ModelLayer>,
  input: Tensor2D,
  context?: LayerForwardContext
): Effect.Effect<Tensor2D, ShapeError> => {
  if (hasSyncNetwork(layers)) {
    return Effect.try({
      try: () => {
        let current = input
        for (const layer of layers) {
          current = layer.forwardSync(current, context)
        }
        return current
      },
      catch: (error) => error as ShapeError
    })
  }

  return Effect.gen(function* () {
    let current = input
    for (const layer of layers) {
      if (hasSyncModelLayer(layer)) {
        current = yield* Effect.try({
          try: () => layer.forwardSync(current, context),
          catch: (error) => error as ShapeError
        })
        continue
      }
      current = yield* layer.forward(current, context)
    }
    return current
  })
}

export const runBackwardPass = (
  layers: ReadonlyArray<ModelLayer>,
  dOut: Tensor2D,
  lr: number,
  cacheKey?: LayerCacheKey
): Effect.Effect<Tensor2D, ShapeError> => {
  if (hasSyncNetwork(layers)) {
    return Effect.try({
      try: () => {
        let grad = dOut
        for (let i = layers.length - 1; i >= 0; i--) {
          grad = layers[i]!.backwardSync(grad, lr, cacheKey)
        }
        return grad
      },
      catch: (error) => error as ShapeError
    })
  }

  return Effect.gen(function* () {
    let grad = dOut
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i]!
      if (hasSyncModelLayer(layer)) {
        grad = yield* Effect.try({
          try: () => layer.backwardSync(grad, lr, cacheKey),
          catch: (error) => error as ShapeError
        })
        continue
      }
      grad = yield* layer.backward(grad, lr)
    }
    return grad
  })
}
