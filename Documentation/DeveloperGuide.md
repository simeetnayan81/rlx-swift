# Developer guide

Practical map of **rlx-swift** for people extending or integrating the library.
Normative contracts live in [`design.md`](../design.md); this guide is how to *work with* them day to day.

For **viewing DocC** and **documenting new APIs**, see [README.md](README.md) in this folder.

---

## What you are building on

`rlx-swift` is an **environment and data-collection substrate** on [mlx-swift](https://github.com/ml-explore/mlx-swift):

| Responsibility | In this package |
|----------------|-----------------|
| MDP loop (`reset` / `step` / `close`) | Yes — `RLXCore` |
| Spaces, seeding, registry, type erasure | Yes |
| Wrappers (limits, transforms, validation) | Yes — `RLXWrappers` |
| Reference / debug envs | Yes — `RLXEnvs` |
| Contract tests | Yes — `RLXTesting` |
| Batched collection (sync / async vector) | Yes — `RLXVector` |
| Policies, losses, optimizers, replay buffers | **No** — your code / future algorithms package |

Single-env APIs are **synchronous** `throws`. Concurrency lives at **vector / driver** (`AsyncVectorEnv`), not on every `Environment`.

---

## Module map (dependency direction)

```text
                    ┌─────────────┐
                    │   RLXCore   │  MLX only
                    └──────┬──────┘
               ┌───────────┼───────────┐
               ▼           ▼           ▼
        RLXWrappers   RLXEnvs*    RLXVector*
               │           │           │
               └─────┬─────┘           │
                     ▼                 │
               RLXTesting              │
                     │                 │
                     └────────┬────────┘
                              ▼
                    RandomAgentDemo / your app

* RLXEnvs and RLXVector depend on RLXWrappers for stacks / info keys.
```

| Target | Import when you need… |
|--------|------------------------|
| `RLXCore` | `Environment`, spaces, `Seed` / `PRNG`, `Info`, registry, errors |
| `RLXWrappers` | Time limits, order checks, passive validation, transforms, stats |
| `RLXEnvs` | DummyEnv, CartPole-v1, Pendulum-v1, default registration |
| `RLXTesting` | `checkEnvironment` in unit tests / CI |
| `RLXVector` | `SyncVectorEnv` / `AsyncVectorEnv` for batched rollouts |

Products are separate libraries: depend only on what you use (e.g. core-only apps need not link vector).

---

## Core interaction model

```text
construct env
    → reset(seed:options:) → ResetResult(observation, info)
    → step(action)*        → StepResult(obs, reward: Float, terminated, truncated, info)
    → when terminated || truncated → reset again (unless vector autoreset)
    → close()              → further reset/step should throw .closed
```

### Rules you must not violate

1. **First call is `reset`** (not `step`). Prefer wrapping with `OrderEnforcing` while developing.
2. **Reward is always `Float`** on single-env `StepResult` (MLX float32 alignment).
3. **Do not** call process-global `MLXRandom.seed` inside env or library code — use `reset(seed:)` and explicit `PRNG` / `SplitMix64`.
4. **`terminated` vs `truncated`**: task end vs external cutoff (e.g. `TimeLimit`). Do not invent a single `done` flag in new public APIs; use `StepResult.done` only as a convenience.
5. **Spaces do not store seeds** — callers pass RNG/key on each `sample`.

See DocC *Environment lifecycle* (`RLXCore`) and `design.md` §11–§12.

---

## Patterns you will use constantly

### 1. Minimal custom environment

Implement `Environment` with matching space `Value` types, lifecycle errors, finite `Float` rewards.
Full walkthrough: DocC **Implement a custom environment** (`RLXWrappers`), source at
`Sources/RLXWrappers/RLXWrappers.docc/Articles/CustomEnvironmentGuide.md`.

### 2. Recommended developer stack

```swift
import RLXCore
import RLXWrappers

let env = PassiveEnvChecker(
    OrderEnforcing(
        TimeLimit(MyEnv(), maxEpisodeSteps: 200)
    )
)
```

| Layer | Catches |
|-------|---------|
| `PassiveEnvChecker` | Obs/action ∉ space, non-finite reward |
| `OrderEnforcing` | `step` before `reset`, `step` after episode end |
| `TimeLimit` | Forces `truncated` at step budget |

Cost of passive checks is **medium** — fine for dev; measure before leaving on ultra-hot collection.

### 3. Registry / make by id

```swift
import RLXEnvs
import RLXCore

try RLXEnvsRegistration.registerDefaults()
let env = try EnvironmentRegistry.shared.make("CartPole-v1")
```

Factories typically install a default wrapper stack (order + time limit) for classic control.
Ids and specs: `design.md` §24; registration in `Sources/RLXEnvs/Registration.swift`.

### 4. Random policy (no algorithms)

```swift
var rng = SplitMix64(seed: 0)
_ = try env.reset(seed: 42 as UInt64?, options: nil)
var done = false
while !done {
    let action = env.actionSpace.sample(using: &rng)
    let step = try env.step(action)
    done = step.done
}
try env.close()
```

Runnable: `swift run RandomAgentDemo`.

### 5. Vector collection

```swift
import RLXVector

let vec = SyncVectorEnv(numEnvs: 8, autoresetMode: .sameStep) {
    AnyEnvironment(DummyEnv(episodeLength: 50))
}
_ = try vec.reset(seed: 1)
let batch = try vec.step(actions)  // length == numEnvs
// sameStep: live obs may be next episode; terminal obs in info[final_observation]
```

Async: `AsyncVectorEnv` (`async throws`, actor-isolated, `maxConcurrency: 1` for serial tests).
Seeding: slot `i` gets `Seed(base).child(index: i)`.

### 6. Contract tests

```swift
import RLXTesting

try checkEnvironment({ MyEnv() })
try checkEnvironment(
    { OrderEnforcing(MyEnv()) },
    options: CheckEnvironmentOptions(episodes: 20, enforceOrder: true)
)
```

Requires `Observation: Equatable` for determinism checks. Use a **factory** so two instances can be built.

---

## Error types (where to look)

| Type | Module | Use |
|------|--------|-----|
| `EnvironmentError` | `RLXCore` | Single-env lifecycle / validation / close |
| `RegistryError` | `RLXCore` | Register / make failures |
| `VectorEnvironmentError` | `RLXVector` | Closed vector, cancelled async work, batch size |
| `CheckEnvironmentError` | `RLXTesting` | Harness failures (not thrown by envs in production) |

Prefer these over ad-hoc `NSError` / stringly errors in library code.

---

## Info keys (stable diagnostics)

Use `InfoKeys` (`RLXWrappers`) — do not invent colliding string literals:

| Constant | Typical setter |
|----------|----------------|
| `timeLimitTruncated` | `TimeLimit` |
| `finalObservation` / `finalInfo` | Vector `sameStep` autoreset |
| `episode` / `episodeReturn` / `episodeLength` / `episodeTime` | `RecordEpisodeStatistics` |

Env-specific keys should be scoped (nested or prefixed) — `design.md` §14.

---

## Seeding cheat sheet

| Goal | Approach |
|------|----------|
| Reproducible episode | `reset(seed: some UInt64, options: nil)` |
| Typed seed values | `Seed(rawValue:)` / `Seed(42)` + `reset(seed: seed)` convenience |
| Vector per-slot | `Seed(base).child(index: i)` (done inside vector `reset`) |
| Sample actions on CPU | `SplitMix64` + `space.sample(using:)` |
| Tensor RNG in dynamics | `PRNG` / `MLXRandom` **explicit keys** only |

---

## Layout of the repository (for PRs)

```text
Sources/RLXCore/          Core contracts + spaces
Sources/RLXWrappers/      Wrappers + InfoKeys
Sources/RLXEnvs/          Reference envs + Registration
Sources/RLXTesting/       checkEnvironment
Sources/RLXVector/        Sync / async vector
Sources/*/…/*.docc/       DocC catalogs (hand-authored)
Examples/RandomAgentDemo/ Minimal runnable loop
Tests/                    XCTest mirrors targets
Documentation/            Developer + DocC workflow guides
design.md                 Normative design
scripts/                  pre-commit, xcodebuild-test, generate-docs, Linux smoke
```

### Adding a feature (checklist)

1. Implement in the correct target (see module map).
2. Public API: accurate `///` (DocC symbol pages).
3. If the behaviour is a **contract**, update `design.md` in the same PR.
4. Unit tests under `Tests/<Target>Tests/`; env contracts via `checkEnvironment` where applicable.
5. Smoke path (`RLXCoreSmoke`) if Linux / CLI must exercise the path without Metal.
6. New conceptual surface → DocC article + Topics link (see [README.md](README.md)).
7. Prefer `swift build`, `swift run RLXCoreSmoke`, `./scripts/xcodebuild-test.sh` before push.

Pre-commit (optional, after `./scripts/install-git-hooks.sh`) runs lint checks, build, smoke, and unit tests.

---

## Platform notes for developers

| Tier | Expectation |
|------|-------------|
| **macOS + Xcode** | Primary: Metal MLX, full XCTest, DocC |
| **Linux** | Tier-2: CPU mlx-swift; gate is build + `RLXCoreSmoke` (avoid Metal-only paths in smoke) |
| **iOS / others** | Declared compile targets; not full test matrix in v1 |

DummyEnv / `Int` spaces are safe for Linux smoke. CartPole / Pendulum / `MLXArray` need macOS tests for full coverage.

---

## Where to read next

| Goal | Start here |
|------|------------|
| Locked semantics | [`design.md`](../design.md) §6–§8, §11–§16, §20–§21, §27 |
| API browser | Xcode **Product → Build Documentation** (see [README.md](README.md)) |
| First custom env | DocC *Implement a custom environment* |
| Validation cost tiers | DocC *Validation layers* (`RLXWrappers`) |
| Vector autoreset | DocC `RLXVector` + `AutoresetMode` |
| Runnable loop | `swift run RandomAgentDemo` |

---

## Out of scope (do not land here without design update)

- Training algorithms (PPO, DQN, …) and optimizers
- Rollout / replay buffers as core products
- Making every `Environment` method `async`
- Subprocess / multi-node workers (future phase)
