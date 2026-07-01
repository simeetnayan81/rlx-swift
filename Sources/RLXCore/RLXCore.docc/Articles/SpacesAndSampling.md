# Spaces and sampling

## Overview

A ``Space`` describes the set of valid values for observations or actions and how to **sample**
them. Concrete spaces include ``DiscreteSpace``, ``BoxSpace``, ``MultiDiscreteSpace``,
``MultiBinarySpace``, ``DictSpace``, and ``TupleSpace``.

### Dual sampling APIs

Every space implements:

- `sample(using:)` — Swift ``RandomNumberGenerator`` path (normative RNG: ``SplitMix64``)
- `sample(key:)` — MLXRandom key path (`MLXArray` key from ``PRNG``)

Spaces do **not** store a seed. Callers own RNG state.

### Contains

`contains(_:)` is the membership test used by ``PassiveEnvChecker`` (`RLXWrappers`) and
``checkEnvironment`` (`RLXTesting`). Implement it accurately for custom spaces.

### Type erasure and flattening

- ``AnySpace`` hides associated `Value` types for registries and mixed stacks.
- ``SpaceFlatten`` maps structured spaces to a dense vector layout for shared MLPs
  (one-hot discrete segments, etc.).

### Related design sections

`design.md` §8.1, §9 (spaces), §13 (PRNG policy).
