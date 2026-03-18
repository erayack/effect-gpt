import * as Effect from "effect/Effect"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"
import * as Clock from "effect/Clock"
import * as Ref from "effect/Ref"
import * as HashMap from "effect/HashMap"

export interface Counter {
  readonly inc: (n?: number) => Effect.Effect<void>
  readonly get: () => Effect.Effect<number>
}

export interface Gauge {
  readonly set: (value: number) => Effect.Effect<void>
  readonly get: () => Effect.Effect<number>
}

export interface Histogram {
  readonly observe: (value: number) => Effect.Effect<void>
  readonly getStats: () => Effect.Effect<{ count: number; sum: number; min: number; max: number; mean: number }>
}

export interface TimingResult<A> {
  readonly value: A
  readonly durationMs: number
}

export interface MetricsService {
  readonly counter: (name: string) => Effect.Effect<Counter>
  readonly gauge: (name: string) => Effect.Effect<Gauge>
  readonly histogram: (name: string) => Effect.Effect<Histogram>
  readonly timed: <A, E, R>(label: string, effect: Effect.Effect<A, E, R>) => Effect.Effect<TimingResult<A>, E, R>
  readonly snapshot: () => Effect.Effect<MetricsSnapshot>
}

export interface MetricsSnapshot {
  readonly counters: ReadonlyArray<{ name: string; value: number }>
  readonly gauges: ReadonlyArray<{ name: string; value: number }>
  readonly histograms: ReadonlyArray<{
    name: string
    count: number
    sum: number
    min: number
    max: number
    mean: number
  }>
  readonly timings: ReadonlyArray<{ label: string; durationMs: number }>
}

class MetricsTag extends Context.Tag("effect-gpt/services/Metrics")<MetricsTag, MetricsService>() {}

export const Metrics = MetricsTag
export type MetricsServiceId = MetricsTag

interface InMemoryState {
  counters: HashMap.HashMap<string, Ref.Ref<number>>
  gauges: HashMap.HashMap<string, Ref.Ref<number>>
  histograms: HashMap.HashMap<string, Ref.Ref<Array<number>>>
  timings: Ref.Ref<Array<{ label: string; durationMs: number }>>
}

const makeInMemoryMetrics = (): Effect.Effect<MetricsService> =>
  Effect.gen(function* () {
    const stateRef = yield* Ref.make<InMemoryState>({
      counters: HashMap.empty(),
      gauges: HashMap.empty(),
      histograms: HashMap.empty(),
      timings: yield* Ref.make<Array<{ label: string; durationMs: number }>>([])
    })

    const getOrCreateCounter = (name: string): Effect.Effect<Ref.Ref<number>> =>
      Effect.gen(function* () {
        const state = yield* Ref.get(stateRef)
        const existing = HashMap.get(state.counters, name)
        if (existing._tag === "Some") {
          return existing.value
        }
        const newRef = yield* Ref.make(0)
        yield* Ref.update(stateRef, (s) => ({
          ...s,
          counters: HashMap.set(s.counters, name, newRef)
        }))
        return newRef
      })

    const getOrCreateGauge = (name: string): Effect.Effect<Ref.Ref<number>> =>
      Effect.gen(function* () {
        const state = yield* Ref.get(stateRef)
        const existing = HashMap.get(state.gauges, name)
        if (existing._tag === "Some") {
          return existing.value
        }
        const newRef = yield* Ref.make(0)
        yield* Ref.update(stateRef, (s) => ({
          ...s,
          gauges: HashMap.set(s.gauges, name, newRef)
        }))
        return newRef
      })

    const getOrCreateHistogram = (name: string): Effect.Effect<Ref.Ref<Array<number>>> =>
      Effect.gen(function* () {
        const state = yield* Ref.get(stateRef)
        const existing = HashMap.get(state.histograms, name)
        if (existing._tag === "Some") {
          return existing.value
        }
        const newRef = yield* Ref.make<Array<number>>([])
        yield* Ref.update(stateRef, (s) => ({
          ...s,
          histograms: HashMap.set(s.histograms, name, newRef)
        }))
        return newRef
      })

    const service: MetricsService = {
      counter: (name) =>
        Effect.gen(function* () {
          const ref = yield* getOrCreateCounter(name)
          return {
            inc: (n = 1) => Ref.update(ref, (v) => v + n),
            get: () => Ref.get(ref)
          }
        }),

      gauge: (name) =>
        Effect.gen(function* () {
          const ref = yield* getOrCreateGauge(name)
          return {
            set: (value) => Ref.set(ref, value),
            get: () => Ref.get(ref)
          }
        }),

      histogram: (name) =>
        Effect.gen(function* () {
          const ref = yield* getOrCreateHistogram(name)
          return {
            observe: (value) => Ref.update(ref, (arr) => [...arr, value]),
            getStats: () =>
              Effect.gen(function* () {
                const values = yield* Ref.get(ref)
                if (values.length === 0) {
                  return { count: 0, sum: 0, min: 0, max: 0, mean: 0 }
                }
                const sum = values.reduce((a, b) => a + b, 0)
                return {
                  count: values.length,
                  sum,
                  min: Math.min(...values),
                  max: Math.max(...values),
                  mean: sum / values.length
                }
              })
          }
        }),

      timed: (label, effect) =>
        Effect.gen(function* () {
          const start = yield* Clock.currentTimeMillis
          const value = yield* effect
          const end = yield* Clock.currentTimeMillis
          const durationMs = Number(end - start)
          const state = yield* Ref.get(stateRef)
          yield* Ref.update(state.timings, (arr) => [...arr, { label, durationMs }])
          return { value, durationMs }
        }),

      snapshot: () =>
        Effect.gen(function* () {
          const state = yield* Ref.get(stateRef)

          const counters: Array<{ name: string; value: number }> = []
          for (const [name, ref] of HashMap.entries(state.counters)) {
            const value = yield* Ref.get(ref)
            counters.push({ name, value })
          }

          const gauges: Array<{ name: string; value: number }> = []
          for (const [name, ref] of HashMap.entries(state.gauges)) {
            const value = yield* Ref.get(ref)
            gauges.push({ name, value })
          }

          const histograms: Array<{
            name: string
            count: number
            sum: number
            min: number
            max: number
            mean: number
          }> = []
          for (const [name, ref] of HashMap.entries(state.histograms)) {
            const values = yield* Ref.get(ref)
            if (values.length > 0) {
              const sum = values.reduce((a, b) => a + b, 0)
              histograms.push({
                name,
                count: values.length,
                sum,
                min: Math.min(...values),
                max: Math.max(...values),
                mean: sum / values.length
              })
            }
          }

          const timings = yield* Ref.get(state.timings)

          return { counters, gauges, histograms, timings }
        })
    }

    return service
  })

export const InMemoryMetricsLive: Layer.Layer<MetricsServiceId> = Layer.effect(Metrics, makeInMemoryMetrics())

const noOpCounter: Counter = {
  inc: () => Effect.void,
  get: () => Effect.succeed(0)
}

const noOpGauge: Gauge = {
  set: () => Effect.void,
  get: () => Effect.succeed(0)
}

const noOpHistogram: Histogram = {
  observe: () => Effect.void,
  getStats: () => Effect.succeed({ count: 0, sum: 0, min: 0, max: 0, mean: 0 })
}

const noOpMetrics: MetricsService = {
  counter: () => Effect.succeed(noOpCounter),
  gauge: () => Effect.succeed(noOpGauge),
  histogram: () => Effect.succeed(noOpHistogram),
  timed: (_, effect) => Effect.map(effect, (value) => ({ value, durationMs: 0 })),
  snapshot: () =>
    Effect.succeed({
      counters: [],
      gauges: [],
      histograms: [],
      timings: []
    })
}

export const NoOpMetricsLive: Layer.Layer<MetricsServiceId> = Layer.succeed(Metrics, noOpMetrics)

export const counter = (name: string) => Effect.flatMap(Metrics, (metrics) => metrics.counter(name))

export const gauge = (name: string) => Effect.flatMap(Metrics, (metrics) => metrics.gauge(name))

export const histogram = (name: string) => Effect.flatMap(Metrics, (metrics) => metrics.histogram(name))

export const timed = <A, E, R>(label: string, effect: Effect.Effect<A, E, R>) =>
  Effect.flatMap(Metrics, (metrics) => metrics.timed(label, effect))

export const snapshot = () => Effect.flatMap(Metrics, (metrics) => metrics.snapshot())
