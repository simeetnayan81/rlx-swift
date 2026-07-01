# Seeding and PRNG

## Overview

Reproducibility is **explicit**:

- Prefer `reset(seed: UInt64?)` on environments.
- Use ``Seed`` as a typed `UInt64` newtype when propagating seeds.
- Use ``PRNG`` as a thin wrapper over mlx-swift `MLXRandom.key` / `split` — not a second algorithm.
- Use ``SplitMix64`` for CPU / `sample(using:)` paths.

### Do not

- Call process-global `MLXRandom.seed` inside library or environment code (cross-talk between
  envs and tests).
- Rely on an undocumented global seed for env dynamics.

### Vector fan-out

Vector envs (`RLXVector`) derive per-slot seeds with ``Seed/child(index:)`` so slot `i`
matches a single-env `reset(seed: child.rawValue)`.

### Related design sections

`design.md` §13 (seeding & PRNG), §27 (locked decisions).
