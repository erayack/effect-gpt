import * as Effect from "effect/Effect"
import * as Context from "effect/Context"
import * as Layer from "effect/Layer"

export interface RandomService {
  readonly next: () => Effect.Effect<number>
  readonly nextGaussian: (mean: number, std: number) => Effect.Effect<number>
  readonly nextInt: (min: number, max: number) => Effect.Effect<number>
  readonly fork: () => Effect.Effect<RandomService>
}

class RandomTag extends Context.Tag("effect-gpt/services/Random")<RandomTag, RandomService>() {}

export const Random = RandomTag
export type RandomServiceId = RandomTag

interface RngState {
  state: number
}

const mulberry32Next = (rng: RngState): number => {
  rng.state = (rng.state + 0x6d2b79f5) >>> 0
  let t = rng.state
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const boxMullerGaussian = (rng: RngState, mean: number, std: number): number => {
  const u1 = mulberry32Next(rng)
  const u2 = mulberry32Next(rng)
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
  return z0 * std + mean
}

const makeSeededRandom = (seed: number): RandomService => {
  const rng: RngState = { state: seed >>> 0 }

  const service: RandomService = {
    next: () => Effect.sync(() => mulberry32Next(rng)),

    nextGaussian: (mean, std) => Effect.sync(() => boxMullerGaussian(rng, mean, std)),

    nextInt: (min, max) =>
      Effect.sync(() => {
        const range = max - min
        return Math.floor(mulberry32Next(rng) * range) + min
      }),

    fork: () =>
      Effect.sync(() => {
        const forkSeed = Math.floor(mulberry32Next(rng) * 0xffffffff)
        return makeSeededRandom(forkSeed)
      })
  }

  return service
}

const makeSystemRandom = (): RandomService => {
  const service: RandomService = {
    next: () => Effect.sync(() => Math.random()),

    nextGaussian: (mean, std) =>
      Effect.sync(() => {
        const u1 = Math.random()
        const u2 = Math.random()
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
        return z0 * std + mean
      }),

    nextInt: (min, max) =>
      Effect.sync(() => {
        const range = max - min
        return Math.floor(Math.random() * range) + min
      }),

    fork: () => Effect.succeed(makeSystemRandom())
  }

  return service
}

export const SeededRandomLive = (seed: number): Layer.Layer<RandomServiceId> =>
  Layer.succeed(Random, makeSeededRandom(seed))

export const SystemRandomLive: Layer.Layer<RandomServiceId> = Layer.succeed(Random, makeSystemRandom())

export const next = () => Effect.flatMap(Random, (random) => random.next())

export const nextGaussian = (mean: number, std: number) =>
  Effect.flatMap(Random, (random) => random.nextGaussian(mean, std))

export const nextInt = (min: number, max: number) =>
  Effect.flatMap(Random, (random) => random.nextInt(min, max))

export const fork = () => Effect.flatMap(Random, (random) => random.fork())
