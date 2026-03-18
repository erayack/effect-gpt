import * as Context from "effect/Context"
import * as Layer from "effect/Layer"
import * as Effect from "effect/Effect"
import { seeded, systemRng, type Rng } from "../tensor/random"

export interface SeedService {
  /** Shared RNG stream used for deterministic initialization. */
  readonly rng: Rng
  /** Creates a new SeedService derived from the current stream (for isolation when desired). */
  readonly fork: () => SeedService
}

class SeedTag extends Context.Tag("effect-gpt/services/Seed")<SeedTag, SeedService>() {}

export const Seed = SeedTag
export type SeedServiceId = SeedTag

const makeSeedService = (seed?: number): SeedService => {
  const rng = seed === undefined ? systemRng() : seeded(seed)

  return {
    rng,
    fork: () => {
      // Advance the stream to produce a derived seed; fall back to system entropy when non-deterministic.
      const nextSeed = Math.floor(rng.next() * 0xffffffff)
      return makeSeedService(seed === undefined ? undefined : nextSeed)
    }
  }
}

/**
 * Layer that provides a shared SeedService. Passing a seed yields deterministic initialization; omitting it uses
 * nondeterministic Math.random.
 */
export const SeedLayer = (seed?: number): Layer.Layer<SeedServiceId> => Layer.succeed(Seed, makeSeedService(seed))

/** Effect helper to grab the current RNG from context. */
export const useSeedRng = (): Effect.Effect<Rng, never, SeedServiceId> => Effect.map(Seed, (service) => service.rng)
