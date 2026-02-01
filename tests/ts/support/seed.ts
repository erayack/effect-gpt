/**
 * Test helper to create seeded RNG with canonical test seed.
 */
import { seeded, type Rng } from "../../../src/tensor/random"

/** Canonical seed for deterministic tests (matches Rust convention). */
export const CANONICAL_SEED = 1337

/** Creates a seeded RNG with the canonical test seed. */
export const testRng = (): Rng => seeded(CANONICAL_SEED)

/** Creates a seeded RNG with a custom seed. */
export const testRngWithSeed = (seed: number): Rng => seeded(seed)
