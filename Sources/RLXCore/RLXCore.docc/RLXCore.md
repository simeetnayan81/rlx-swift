# ``RLXCore``

Reinforcement learning **environment and data-collection substrate** for Swift, built on
[mlx-swift](https://github.com/ml-explore/mlx-swift).

@Metadata {
    @TechnologyRoot
}

## Overview

`RLXCore` defines the **MDP interaction contracts** that the rest of `rlx-swift` builds on:

- ``Environment`` — synchronous `reset` / `step` / `close`
- ``Space`` — observation and action sets (`contains`, dual sampling APIs)
- ``StepResult`` / ``ResetResult`` — transition outcomes with ``Float`` rewards
- ``Info`` — structured side-channel diagnostics
- ``Seed`` / ``PRNG`` / ``SplitMix64`` — explicit, reproducible randomness
- ``EnvironmentRegistry`` — register and `make` envs by id
- ``AnyEnvironment`` / ``AnySpace`` — type erasure for heterogeneous stacks

It depends on the **MLX** product only. Policies, losses, optimizers, and replay buffers are
**out of scope** (use `MLXNN` / `MLXOptimizers` / a future algorithms package).

The normative design document in the repository root, **`design.md`**, is the source of truth
for contracts. DocC and code comments implement and teach those contracts; they do not replace them.

### What this package is not

| Out of scope in `rlx-swift` | Lives elsewhere |
|-----------------------------|-----------------|
| PPO, DQN, SAC, GAE | Future `rlx-swift-algorithms` / your training code |
| Rollout / replay buffers | Same |
| Multi-node cluster orchestration | Future work |
| `async` on every single env | Async only at ``AsyncVectorEnv`` / drivers (`RLXVector`) |

## Topics

### Essentials

- <doc:EnvironmentLifecycle>
- <doc:SpacesAndSampling>
- <doc:SeedingAndPRNG>
- <doc:InfoAndSpecs>

### Protocols and results

- ``Environment``
- ``Space``
- ``StepResult``
- ``ResetResult``
- ``Renderable``

### Spaces

- ``DiscreteSpace``
- ``BoxSpace``
- ``MultiDiscreteSpace``
- ``MultiBinarySpace``
- ``DictSpace``
- ``TupleSpace``
- ``AnySpace``
- ``SpaceFlatten``

### Identity and diagnostics

- ``Info``
- ``InfoValue``
- ``EnvSpec``
- ``EnvironmentError``
- ``RLXCore``

### Seeding

- ``Seed``
- ``PRNG``
- ``SplitMix64``
- ``EnvPRNGStreams``

### Registry and erasure

- ``EnvironmentRegistry``
- ``AnyEnvironment``
- ``ResetOptions``
- ``RenderMode``

## See also

- Repository design document: `design.md` (§6–§14, §20–§21, §27–§28)
- Sibling modules: `RLXWrappers`, `RLXEnvs`, `RLXTesting`, `RLXVector`
