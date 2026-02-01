/**
 * Stub implementations for integration testing.
 */
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import type { Tensor2D } from "../../../src/tensor/Tensor2D"
import * as T from "../../../src/tensor/Tensor2D"
import type { ShapeError } from "../../../src/tensor/ops"
import type { ModelLayer } from "../../../src/model/ModelLayer"
import { SilentLoggerLive } from "../../../src/services/Logger"
import { NoOpMetricsLive } from "../../../src/services/Metrics"

/**
 * Stub OutputProjection that forces EOS token after N forward calls.
 * Used to test prediction loop termination.
 */
export class StubOutputProjection implements ModelLayer {
  readonly _tag = "OutputProjection" as const
  private callCount = 0
  private readonly eosAfter: number
  private readonly eosTokenId: number
  private readonly vocabSize: number

  constructor(vocabSize: number, eosTokenId: number, eosAfter: number) {
    this.vocabSize = vocabSize
    this.eosTokenId = eosTokenId
    this.eosAfter = eosAfter
  }

  get parametersCount(): number {
    return 0
  }

  forward(input: Tensor2D): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.sync(() => {
      this.callCount++
      const logits = T.zeros(input.rows, this.vocabSize)
      const targetToken = this.callCount >= this.eosAfter ? this.eosTokenId : 0
      const lastRowStart = (input.rows - 1) * this.vocabSize
      logits.data[lastRowStart + targetToken] = 100.0
      return logits
    })
  }

  backward(_dOut: Tensor2D, _lr: number): Effect.Effect<Tensor2D, ShapeError> {
    return Effect.succeed(T.zeros(1, 1))
  }

  resetCallCount(): void {
    this.callCount = 0
  }
}

/**
 * Shared test services layer with silent logging and no-op metrics.
 * Use this in tests to suppress log output and avoid metrics overhead.
 */
export const TestServicesLayer = Layer.mergeAll(SilentLoggerLive, NoOpMetricsLive)
