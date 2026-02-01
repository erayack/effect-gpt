/**
 * Deterministic random number generator for reproducible weight initialization.
 * Uses mulberry32 algorithm for fast, seedable pseudo-random numbers.
 */

export interface Rng {
  /** Returns a random number in [0, 1) */
  next(): number
}

/**
 * Creates a seeded RNG using the mulberry32 algorithm.
 * Produces deterministic sequences for reproducible tests.
 */
export const seeded = (seed: number): Rng => {
  let state = seed >>> 0

  return {
    next(): number {
      state = (state + 0x6d2b79f5) >>> 0
      let t = state
      t = Math.imul(t ^ (t >>> 15), t | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }
}

/** Non-deterministic RNG wrapper around Math.random for convenience. */
export const systemRng = (): Rng => ({
  next: () => Math.random()
})
