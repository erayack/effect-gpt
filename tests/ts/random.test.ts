import { describe, expect, it } from "bun:test"
import { seeded } from "../../src/tensor/random"
import { testRng, CANONICAL_SEED } from "./support/seed"

describe("Rng", () => {
  it("produces deterministic sequence with same seed", () => {
    const rng1 = seeded(42)
    const rng2 = seeded(42)

    const seq1 = Array.from({ length: 10 }, () => rng1.next())
    const seq2 = Array.from({ length: 10 }, () => rng2.next())

    expect(seq1).toEqual(seq2)
  })

  it("produces different sequences with different seeds", () => {
    const rng1 = seeded(42)
    const rng2 = seeded(43)

    const seq1 = Array.from({ length: 10 }, () => rng1.next())
    const seq2 = Array.from({ length: 10 }, () => rng2.next())

    expect(seq1).not.toEqual(seq2)
  })

  it("values are in [0, 1) range", () => {
    const rng = testRng()
    for (let i = 0; i < 1000; i++) {
      const val = rng.next()
      expect(val).toBeGreaterThanOrEqual(0)
      expect(val).toBeLessThan(1)
    }
  })

  it("testRng uses canonical seed", () => {
    const rng1 = testRng()
    const rng2 = seeded(CANONICAL_SEED)

    const seq1 = Array.from({ length: 5 }, () => rng1.next())
    const seq2 = Array.from({ length: 5 }, () => rng2.next())

    expect(seq1).toEqual(seq2)
  })
})
