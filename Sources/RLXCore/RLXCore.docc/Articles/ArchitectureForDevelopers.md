# Architecture for developers

How `rlx-swift` is split across modules and how a typical integration is layered.

## Package identity

`rlx-swift` provides the **environment and data-collection substrate** on mlx-swift.
It is **not** an algorithms package: policies, losses, optimizers, and buffers live outside
(e.g. `MLXNN`, `MLXOptimizers`, your training code, future `rlx-swift-algorithms`).

Normative contracts: repository `design.md`. This article is a developer map.

## Targets

| Target | Depends on | Role |
|--------|------------|------|
| ``RLXCore`` | MLX | Protocols, spaces, results, seed/PRNG, registry, errors, type erasure |
| `RLXWrappers` | RLXCore | Lifecycle, limits, transforms, stats, passive validation |
| `RLXEnvs` | RLXCore, RLXWrappers | DummyEnv, CartPole, Pendulum, default registration |
| `RLXTesting` | RLXCore, RLXWrappers | ``checkEnvironment`` harness |
| `RLXVector` | RLXCore, RLXWrappers | Sync and async vector envs |

Depend only on products you need. A pure env implementor often uses **RLXCore + RLXWrappers + RLXTesting**.

## Synchronous core, concurrent edge

- ``Environment``: **sync** `throws` only (`reset` / `step` / `close`).
- Parallelism: ``SyncVectorEnv`` (sequential slots) or ``AsyncVectorEnv`` (`async throws`, actor).
- Rationale: avoid dual APIs and `Sendable` burden on every custom env (`design.md` §21).

## Data flow (single env)

```text
Policy / agent
     │  action (Action space Value)
     ▼
[ Wrappers: PassiveEnvChecker → … → OrderEnforcing ]
     │
     ▼
Concrete Environment (dynamics, PRNG streams)
     │
     ▼
StepResult(observation, reward: Float, terminated, truncated, info)
```

## Data flow (vector)

```text
Batched actions [numEnvs]
     │
     ▼
SyncVectorEnv / AsyncVectorEnv  (AutoresetMode, child seeds)
     │
     ▼
VectorStepResult (per-slot arrays, fixed index order)
```

On ``AutoresetMode/sameStep``, live observations may belong to the **next** episode;
terminal observations are recovered from `info` (`final_observation` / `final_info`).

## Extending the library

1. Prefer new **wrappers** or **envs** over changing core protocols.
2. Document public APIs with `///` (feeds these DocC symbol pages).
3. Contract changes update repository `design.md` in the same PR.
4. Register reference envs via factories that document their wrapper stack.
5. Runnable loop: `Examples/RandomAgentDemo` (`swift run RandomAgentDemo`).

Do not duplicate long recipes here — follow the related articles for each topic.

## Related articles

- <doc:EnvironmentLifecycle>
- <doc:SpacesAndSampling>
- <doc:SeedingAndPRNG>
- <doc:InfoAndSpecs>

In **RLXWrappers** DocC: *Validation layers*, *Wrapper composition*, *Implement a custom environment*.
