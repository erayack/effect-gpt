import { describe, expect, it } from "bun:test"
import { initNormal } from "../../src/tensor/ops"
import { seeded } from "../../src/tensor/random"

describe("initNormal with seeded RNG", () => {
  it("produces deterministic tensors with same seed", () => {
    const rng1 = seeded(42)
    const rng2 = seeded(42)

    const t1 = initNormal(4, 4, 0, 1, rng1)
    const t2 = initNormal(4, 4, 0, 1, rng2)

    expect(Array.from(t1.data)).toEqual(Array.from(t2.data))
  })

  it("produces different tensors with different seeds", () => {
    const rng1 = seeded(42)
    const rng2 = seeded(43)

    const t1 = initNormal(4, 4, 0, 1, rng1)
    const t2 = initNormal(4, 4, 0, 1, rng2)

    expect(Array.from(t1.data)).not.toEqual(Array.from(t2.data))
  })

  it("respects mean and std parameters", () => {
    const rng = seeded(1337)
    const mean = 5
    const std = 0.1
    const t = initNormal(100, 100, mean, std, rng)

    let sum = 0
    for (let i = 0; i < t.data.length; i++) {
      sum += t.data[i]
    }
    const actualMean = sum / t.data.length

    // Mean should be close to target (within ~3 std errors)
    expect(Math.abs(actualMean - mean)).toBeLessThan(0.1)
  })
})
