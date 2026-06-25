# rlx-swift — Design Document

**Status:** Draft v1.2 (source of truth for implementation)  
**Changelog v1.2:** Identity is MLX-native RL infrastructure; dropped external-framework framing. v1.1 locked toolchain/package decisions (§27).  
**Project:** `rlx-swift` — reinforcement learning infrastructure built on [mlx-swift](https://github.com/ml-explore/mlx-swift) (`MLXArray`, Metal/CPU backends, SwiftPM)  
**Audience:** Implementers, reviewers, algorithm authors (PPO / DQN / A2C / etc.)  
**Scope:** Architecture, API contracts, lifecycle semantics, module boundaries. **No implementation in this phase.**

---

## Table of Contents

1. [Purpose & Goals](#1-purpose--goals)
2. [Non-Goals](#2-non-goals)
3. [Design Principles](#3-design-principles)
4. [Positioning: RL on mlx-swift](#4-positioning-rl-on-mlx-swift)
5. [mlx-swift integration](#5-mlx-swift-integration)
6. [Package & Module Layout](#6-package--module-layout)
7. [Core Concepts & Type Model](#7-core-concepts--type-model)
8. [Primary Protocols (API Contracts)](#8-primary-protocols-api-contracts)
9. [Spaces](#9-spaces)
10. [Environment Lifecycle](#10-environment-lifecycle)
11. [Step & Reset Semantics](#11-step--reset-semantics)
12. [Termination vs Truncation](#12-termination-vs-truncation)
13. [Seeding & Determinism](#13-seeding--determinism)
14. [Info, Specs & Metadata](#14-info-specs--metadata)
15. [Wrappers & Transforms](#15-wrappers--transforms)
16. [Vectorized & Parallel Environments](#16-vectorized--parallel-environments)
17. [Rendering](#17-rendering)
18. [Registry & Factory](#18-registry--factory)
19. [Algorithm-Agnostic Integration Surface](#19-algorithm-agnostic-integration-surface)
20. [Error Model & Validation](#20-error-model--validation)
21. [Concurrency & Thread Safety](#21-concurrency--thread-safety)
22. [Platform Strategy](#22-platform-strategy)
23. [Testing Strategy](#23-testing-strategy)
24. [Reference Environments (Bootstrap Set)](#24-reference-environments-bootstrap-set)
25. [Phased Implementation Roadmap](#25-phased-implementation-roadmap)
26. [Key Decisions](#26-key-decisions)
27. [Resolved Questions (formerly open)](#27-resolved-questions-formerly-open--industry-standards)
28. [PR Plan](#28-pr-plan)
29. [Appendix A — Protocol Sketch (Normative Pseudocode)](#appendix-a--protocol-sketch-normative-pseudocode)
30. [Appendix B — Glossary](#appendix-b--glossary)

---

## 1. Purpose & Goals

`rlx-swift` is a **reinforcement learning library for Swift**, implemented on top of **mlx-swift**. It supplies the environment and data-collection substrate that training code (PPO, DQN, A2C, SAC, custom research) builds on: MDP interaction (`reset` / `step`), observation/action spaces, wrappers, vectorized execution, seeding, and episode lifecycle — all with **`MLXArray` as the primary tensor type** for policies and batches on Apple silicon and other MLX backends.

It is the **environment and rollout substrate** for MLX training code. It is **not** an algorithms package: policies, losses, and optimizers live in separate targets/packages (`MLXNN`, `MLXOptimizers`, future `rlx-swift-algorithms`).

### Goals (numbered as specified)

| # | Goal | Design response |
|---|------|-----------------|
| 1 | Clear RL interaction API in idiomatic Swift | `Environment` protocol with `reset` / `step` / `close`; `Space` hierarchy; registry/`make`; wrappers |
| 2 | Strong type safety without blocking dynamic use cases | Primary generics (`Obs`, `Act`) + type-erased `AnyEnvironment` / `AnySpace` for registries and heterogeneous batches |
| 3 | Native MLX integration for tensor-first workloads | Default observation/action encodings as `MLXArray`; batching via MLX; optional Swift scalar/struct paths |
| 4 | Clear episode lifecycle with explicit termination vs truncation | Structured `StepResult`: observation, reward, `terminated`, `truncated`, `info` |
| 5 | Determinism & reproducibility through explicit seeding | `reset(seed:options:)`; `PRNG` / `Seed` types; no global implicit RNG for env dynamics |
| 6 | Composability through wrappers and transforms | `Wrapper` protocol; ordered stack; space remapping; reward/obs/action transforms |
| 7 | Scalable execution: single → vectorized → parallel | `VectorEnvironment`, `AsyncVectorEnv`, batch `MLXArray` layouts |
| 8 | Algorithm-agnostic core | No policy/optimizer/loss in core; buffers/GAE belong in algorithm packages (§19.2) |
| 9 | Cross-platform with Apple-first optimization | SwiftPM package; MLX/Metal path on Apple; CPU/other backends via mlx-swift; macOS tier-1 CI (§27.10) |
| 10 | Testable & predictable behaviour | Deterministic reference envs; contract tests; `checkEnvironment`; reproducible seeds; equation/invariant tests (§27.7) |

---

## 2. Non-Goals

The following are **explicitly out of scope** for the initial design and core package:

1. **Training algorithms** (PPO, DQN, A2C, SAC, etc.) — live in separate packages (e.g. `rlx-swift-algorithms`) or user code.
2. **Rollout / replay buffers & GAE** in v1 — belong with training/algorithm code, not the env core; see §19.2 / §27.9.
3. **Neural network definitions** — use `MLXNN` directly; `rlx-swift` only defines interfaces policies must satisfy if needed later.
4. **Physics engines / game engines** — may appear as optional adapters (`RLXSpriteKit`, `RLXRealityKit`) later; not in core.
5. **Foreign-runtime interop bridges** — optional future packages; not required for v1 and not a design driver.
6. **Distributed multi-node RL** — vector/parallel on a single machine first; cluster orchestration later.
7. **GUI trainer apps** — examples only, not library surface.
8. **Cross-runtime trajectory bit-identity** — v1 targets equation/invariant correctness and on-platform determinism under fixed seeds (§27.7).
9. **`async` methods on every single env** — sync `Environment` only; async at `AsyncVectorEnv` / drivers (§21.1).

---

## 3. Design Principles

1. **MLX-native first.** Design APIs so policies, batches, and spaces work naturally with `MLXArray`, lazy eval, and MLX PRNG keys — not as an afterthought bolted onto a non-tensor model.
2. **Contracts over frameworks.** Protocols define behaviour; concrete types implement; minimal base classes.
3. **Swift-first ergonomics.** Value types for results/specs; `throws` for recoverable errors; structured concurrency for async vector envs.
4. **Tensor-native by default, scalar-friendly when needed.** `MLXArray` is the interchange format for batched RL; single-env APIs may use typed scalars/structs mapped to tensors at boundaries.
5. **Explicit > implicit.** Termination/truncation split; seed on reset; render mode at construction; no silent autoreset unless a vector env documents it.
6. **Composable, not monolithic.** Thin core (`RLXCore`) + spaces + wrappers + vector + optional reference envs.
7. **Standard RL interaction model.** Episodes via `reset`/`step`, spaces for valid obs/actions, wrappers for composition — expressed in Swift/MLX terms, not imported from another ecosystem’s type system.
8. **Fail fast in debug / development; opt-in strict checks in production.** `checkEnvironment`, order-enforcing wrappers, space `contains` checks.
9. **One source of truth for episode state.** After `terminated || truncated`, next legal call is `reset` (unless a documented autoreset vector policy applies).

---

## 4. Positioning: RL on mlx-swift

`rlx-swift` sits in the **mlx-swift ecosystem** the way training utilities sit on a tensor library: it does not reimplement arrays, autodiff, or NN layers. It defines how **agents interact with environments** and how **rollouts are structured for MLX-backed learners**.

| Layer | Responsibility | Package / dependency |
|-------|----------------|----------------------|
| Tensors, devices, random, ops | Numerical substrate | **mlx-swift** (`MLX`) |
| Modules, optimizers | Policies & training steps | **mlx-swift** (`MLXNN`, `MLXOptimizers`) — used by algorithm code, not `RLXCore` |
| Environments, spaces, wrappers, vector envs | RL interaction & collection | **`rlx-swift`** (`RLXCore`, optional targets) |
| PPO / DQN / … | Losses, buffers, update loops | **`rlx-swift-algorithms`** or user apps |

**What rlx-swift owns**

- `Environment` / `Space` / `StepResult` / `ResetResult` contracts
- Seeding and episode lifecycle rules
- Wrappers and registry/`make` for discoverable envs
- Vectorized stepping with batched `MLXArray` layouts
- Small reference envs to validate the stack end-to-end

**What rlx-swift does not own**

- Training algorithms, replay/rollout buffers, or GAE (algorithm layer)
- Neural network modules or optimizers (mlx-swift `MLXNN` / `MLXOptimizers`)
- Compatibility guarantees with non-Swift RL APIs

Core ideas (`reset`/`step`, spaces, termination vs truncation, wrappers, vectorization) are **widely used in modern RL**; rlx adopts them because they work for MLX training loops, not because the library is defined by compatibility with another stack.

---

## 5. mlx-swift integration

mlx-swift is the **foundation** of rlx-swift, not an optional backend.

| mlx-swift capability | Role in rlx-swift |
|----------------------|-------------------|
| `MLXArray` | Primary type for tensor observations, actions, batched rewards/masks |
| `MLX` random (`key`, `split`, `uniform`, …) | Space sampling, env stochasticity, reproducible rollouts without process-global seed mutation |
| `eval(_:)` / lazy execution | Defer materialization until policy/env boundaries need concrete values |
| `DType`, shapes, device/stream | Space metadata and optional placement for vectorized ops |
| `MLXNN.Module` | **Not** a `RLXCore` dependency; algorithm packages compose policies on top |
| `MLXOptimizers` | **Not** a `RLXCore` dependency |
| Platform matrix (macOS, iOS, …) | rlx declares platforms compatible with the **pinned** mlx-swift release |

**Dependency rule:** `RLXCore` depends only on `MLX` (and Foundation/Swift stdlib). It must **not** depend on `MLXNN` or `MLXOptimizers`, keeping the environment layer algorithm-agnostic and lightweight while staying firmly in the MLX stack.

```
┌─────────────────────────────────────────────────────────────┐
│  User / Algorithms (PPO, DQN, …)  — separate package/app    │
│       depends on: RLXCore, RLXEnvs?, MLXNN, MLXOptimizers   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  RLXVector / RLXWrappers / RLXEnvs  (optional layers)       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  RLXCore  (Environment, Space, Step/Reset, Seed, Registry)  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  MLX  (MLXArray, random, ops, device)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Package & Module Layout

**Normative layout:** standalone GitHub repository `rlx-swift` (SwiftPM package), depending on `mlx-swift` via SwiftPM — **not** vendored inside the mlx-swift tree (matches `mlx-swift-lm` / `mlx-swift-examples` pattern: satellite packages around core MLX).

```
rlx-swift/
├── Package.swift
├── design.md                    # this document (may live in planning branch first)
├── Sources/
│   ├── RLXCore/                 # REQUIRED — protocols, spaces, results, seed, errors
│   │   ├── Environment.swift
│   │   ├── Space.swift
│   │   ├── Spaces/              # Box, Discrete, MultiDiscrete, MultiBinary, Dict, Tuple, …
│   │   ├── StepResult.swift
│   │   ├── ResetResult.swift
│   │   ├── Info.swift
│   │   ├── Seed.swift / PRNG.swift
│   │   ├── Spec.swift           # EnvSpec, TimeLimit config, etc.
│   │   ├── Registry.swift
│   │   ├── Errors.swift
│   │   ├── RenderMode.swift
│   │   └── TypeErasure.swift    # AnyEnvironment, AnySpace
│   ├── RLXWrappers/             # optional target
│   │   ├── Wrapper.swift
│   │   ├── TimeLimit.swift
│   │   ├── OrderEnforcing.swift
│   │   ├── PassiveEnvChecker.swift
│   │   ├── ClipAction.swift / RescaleAction.swift
│   │   ├── TransformObservation.swift / TransformReward.swift
│   │   ├── NormalizeObservation.swift / NormalizeReward.swift (running stats)
│   │   └── RecordEpisodeStatistics.swift
│   ├── RLXVector/               # optional target
│   │   ├── VectorEnvironment.swift
│   │   ├── SyncVectorEnv.swift
│   │   ├── AsyncVectorEnv.swift
│   │   └── AutoresetMode.swift
│   ├── RLXEnvs/                 # optional classic / toy envs
│   │   ├── ClassicControl/      # CartPole, MountainCar, Pendulum, Acrobot (tensor-native)
│   │   ├── ToyText/             # optional small discrete envs
│   │   └── Registration.swift   # register defaults into registry
│   └── RLXTesting/              # optional — checkEnvironment, mocks, determinism helpers
├── Tests/
│   ├── RLXCoreTests/
│   ├── RLXWrappersTests/
│   ├── RLXVectorTests/
│   └── RLXEnvsTests/
└── Examples/                    # optional executable targets
    └── RandomAgentDemo/
```

### Target dependency graph

```
RLXCore        → MLX
RLXWrappers    → RLXCore
RLXVector      → RLXCore, RLXWrappers (optional link)
RLXEnvs        → RLXCore, RLXWrappers
RLXTesting     → RLXCore, RLXWrappers
```

Users who only need contracts + custom envs link **`RLXCore` only**.

---

## 7. Core Concepts & Type Model

### 7.1 Observation & Action types

Every environment is parameterized by observation and action **element types**:

```text
Environment where Observation == …, Action == …
```

**Recommended defaults for MLX-centric workloads:**

| Role | Preferred type | Notes |
|------|----------------|-------|
| Flat continuous obs/act | `MLXArray` | Shape fixed by `Box` space |
| Discrete action | `Int` or `MLXArray` (scalar/int32) | Document per env |
| Structured obs | `Observation` struct + `DictSpace` / custom encoder | Encode to `MLXArray` via `TensorConvertible` |
| Batched (vector env) | `MLXArray` with leading batch dim | `shape[0] == numEnvs` |

### 7.2 Dual API layers

To satisfy goal #2 (type safety + dynamic use cases):

1. **Typed layer (primary):** `Environment`, `Space` with associated types — compile-time safety for custom envs and algorithms written against concrete envs.
2. **Type-erased layer:** `AnyEnvironment`, `AnySpace` — **manual type-eraser classes** (Swift/Combine/`AnyPublisher` pattern; not solely `any Environment` existentials). See §27.4. Used for registry storage and `make(id:)`.
3. **Tensor layer:** `TensorEnvironment` / adapters that always speak `MLXArray` for obs/act/reward — optimal for vectorized training loops.

### 7.3 Result value types

All step/reset outcomes are **structs** (value semantics), not tuples only — named fields, `Equatable` where feasible, easy to log/test.

### 7.3.1 Reward scalar type (locked)

| Layer | Type | Rationale |
|-------|------|-----------|
| Single-env `StepResult.reward` | `Float` (IEEE-754 binary32) | Matches MLX / deep-RL training default (float32). Fixing `Float` avoids associated-type explosion on every env/wrapper. |
| Vector-env rewards | `MLXArray` shape `[numEnvs]`, dtype `.float32` | Batch training interchange; convertible from `[Float]`. |
| Logging / human metrics | May promote to `Double` at boundaries only | Not part of core step contract. |

**Non-goal:** per-environment `associatedtype Reward`. If a niche env needs higher precision internally, it quantizes/casts to `Float` at the `step` boundary (same pattern as casting obs to space dtype).

### 7.4 Associated types pattern

```swift
public protocol Environment: AnyObject {
    associatedtype Observation
    associatedtype Action
    associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation
    associatedtype ActionSpace: Space where ActionSpace.Value == Action

    var observationSpace: ObservationSpace { get }
    var actionSpace: ActionSpace { get }
    // …
}
```

`AnyObject` (class-bound) is recommended for environments because they hold mutable episode state; spaces may be structs.

---

## 8. Primary Protocols (API Contracts)

This section is the **normative API surface** implementers must honor. Appendix A expands to full method signatures.

### 8.1 `Space`

Defines the set of valid values for observations or actions.

**Requirements:**

| Member | Semantics |
|--------|-----------|
| `associatedtype Value` | Swift type of a single sample |
| `sample(using rng: inout some RandomNumberGenerator) -> Value` | Uniform (or space-defined) random valid value |
| `sample(key: MLXArray) -> Value` | MLX-keyed sample for GPU/reproducible tensor pipelines |
| `contains(_ value: Value) -> Bool` | Membership test |
| `shape: [Int]?` | Tensor shape if applicable (`nil` for non-tensor spaces) |
| `dtype: DType?` | MLX dtype if tensor-backed |
| `seededSample` / split keys | Must not mutate global MLX seed unless explicitly documented |

**Concrete spaces (v1 minimum set):**

| Space | `Value` (typical) | Role |
|-------|-------------------|------|
| `DiscreteSpace` | `Int` | Finite action/obs index set `{0..<n}` |
| `BoxSpace` | `MLXArray` | Continuous (or quantized) tensor region with bounds |
| `MultiDiscreteSpace` | `[Int]` or `MLXArray` | Product of discrete sets |
| `MultiBinarySpace` | `MLXArray` (0/1) | Binary vectors |
| `TupleSpace` | heterogeneous tuple via type erasure | Fixed-arity heterogeneous product |
| `DictSpace` | keyed structured obs/act | Named fields; ordered keys for flatten |
| `TextSpace` (optional v1.1) | `String` | Token/text observations |

**Operations on spaces:**

- `seed` is **not** stored on the space long-term in the preferred design; callers pass `PRNG`/`key` into `sample`. Optional `SeededSpace` wrapper caches a key for convenience.
- `flatten` / `unflatten` utilities for converting structured spaces to a single `Box`/`MLXArray` (common for NN policies).

### 8.2 `Environment`

Core MDP / POMDP interaction surface.

| Member | Required | Semantics |
|--------|----------|-----------|
| `id` / `spec` | recommended | Stable string id + `EnvSpec` metadata |
| `observationSpace` | yes | Valid observations |
| `actionSpace` | yes | Valid actions |
| `reset(seed:options:)` | yes | Start or restart episode; see §11 |
| `step(_:)` | yes | Transition; see §11 |
| `close()` | yes | Release resources (idempotent) |
| `render()` | optional via `Renderable` | See §17 |
| `unwrapped` | via wrapper protocol | Traverse to base env |

**Invariants (must hold):**

1. First interaction must be `reset` before `step` (enforceable via `OrderEnforcing` wrapper).
2. After `step` returns `terminated || truncated == true`, caller must `reset` before another `step` (same enforcement).
3. `observation` from `reset`/`step` must satisfy `observationSpace.contains`.
4. `action` passed to `step` should satisfy `actionSpace.contains` (checked in debug/checker wrappers).
5. `close()` may be called multiple times; subsequent ops should throw `EnvironmentError.closed`.

### 8.3 `Wrapper` / `EnvironmentWrapper`

```text
Wrapper: Environment where inner: some Environment
```

- Forwards unknown behaviour to `inner`.
- May override `reset`, `step`, spaces, `render`, `close`.
- Must preserve termination/truncation semantics unless the wrapper’s contract explicitly changes episode boundaries (e.g. `TimeLimit` sets `truncated`).
- `unwrapped` returns innermost non-wrapper env.

### 8.4 `VectorEnvironment`

Batched parallel (logical) environments with **one** `reset`/`step` over `N` sub-environments.

| Member | Semantics |
|--------|-----------|
| `numEnvs: Int` | Batch size |
| `observationSpace` / `actionSpace` | Single-env spaces (batching is external layout) |
| `singleObservationSpace` / `singleActionSpace` | aliases for clarity |
| `reset(seed:options:)` | Returns batched obs + infos |
| `step(actions:)` | Batched actions → batched transitions |
| `autoresetMode` | See §16 |

### 8.5 `PRNG` / `Seedable`

```text
Seed: UInt64 (or struct wrapping UInt64 + stream id)
PRNG: wraps MLX key (MLXArray) and/or Swift RNG
```

- Environments that need randomness implement `SeedableEnvironment` or accept seed only via `reset(seed:)`.
- **No** required global `seed()` method on env; seeding happens via `reset(seed:)` (and explicit space/PRNG APIs).

### 8.6 `EnvSpec`

Immutable metadata for an environment kind (not instance state):

- `id: String` — e.g. `"CartPole-v1"`
- `maxEpisodeSteps: Int?`
- `rewardThreshold: Double?` (solved criterion, optional)
- `nondeterministic: Bool`
- `defaultRenderMode: RenderMode?`
- `version: Int`
- `kwargs` / config schema description

### 8.7 `Registry`

```text
EnvironmentRegistry.shared.register(id:factory:)
EnvironmentRegistry.shared.make(id:config:renderMode:) throws -> AnyEnvironment
```

- Factories are `@Sendable` closures or protocol existentials producing type-erased envs.
- Collision policy: last register wins **or** throw on duplicate (configurable; default **throw** in debug, documented override for tests).

### 8.8 Supporting protocols

| Protocol | Role |
|----------|------|
| `Renderable` | `render() throws -> RenderFrame?` |
| `Closeable` | `close()` — env conforms |
| `Configurable` | `associatedtype Config`; init from config |
| `TensorEncodable` | `func asMLXArray() -> MLXArray` |
| `TensorDecodable` | `init(mlxArray:space:)` |
| `EnvironmentChecker` | Static/runtime validation (`checkEnvironment`) |

---

## 9. Spaces

### 9.1 Design rules

1. Spaces are **pure descriptions + sampling/membership** — they do not own episode state.
2. Tensor spaces (`Box`, `MultiBinary`) use `MLXArray` values with documented shape/dtype.
3. Sampling must support:
   - Swift `RandomNumberGenerator` (CPU, easy testing)
   - MLX `key: MLXArray` (training reproducibility on device)
4. `contains` should be side-effect free and reasonably fast (used in checkers).
5. Equality of spaces: same parameters (bounds, shape, n) ⇒ equal; used in tests/wrappers.

### 9.2 `BoxSpace`

- Parameters: `low: MLXArray`, `high: MLXArray`, `shape: [Int]`, `dtype: DType` (default `.float32`).
- Broadcast-compatible bounds (scalar low/high allowed, stored expanded or lazy).
- Unbounded: use `±infinity` explicitly; document whether bounds are hard constraints or soft limits.
- `sample`: uniform in `[low, high]` elementwise (discrete dtypes use integer uniform).

### 9.3 `DiscreteSpace`

- Parameters: `n: Int` (values `0 ..< n`), optional `start: Int` (default 0) when the index set is offset.
- `sample` → `Int` in range.
- Optional tensor mode: return `MLXArray` scalar int32 for vectorized policies.

### 9.4 Composite spaces

- `DictSpace`: ordered keys for deterministic flatten order.
- `TupleSpace`: fixed arity.
- Flatten utilities produce a single vector `MLXArray` + metadata to unflatten — critical for shared MLP policies.

### 9.5 Space transforms (wrappers may expose new spaces)

Examples:

- `RescaleAction`: maps policy `[-1,1]` box to env box.
- `FlattenObservation`: `Dict`/`Tuple` → `Box`.
- `DiscreteToBox`: one-hot / embedded actions (algorithm-side more often than wrapper).

---

## 10. Environment Lifecycle

State machine for a **single** environment instance:

```text
                    ┌──────────────┐
                    │  uninitialized│
                    └──────┬───────┘
                           │ reset(seed?, options?)
                           ▼
                    ┌──────────────┐
         ┌─────────│   running    │◄────────────┐
         │         └──────┬───────┘             │
         │                │ step(action)        │ reset(...)
         │                ▼                     │
         │         ┌──────────────┐             │
         │         │ step returned│             │
         │         └──────┬───────┘             │
         │      terminated│or truncated?        │
         │         no     │ yes                 │
         │         └──────┘                     │
         │                │                     │
         │                ▼                     │
         │         ┌──────────────┐             │
         │         │ episode_end  │─────────────┘
         │         └──────────────┘   (must reset before step)
         │
         │ close()
         ▼
  ┌──────────────┐
  │    closed    │  (terminal; close is idempotent)
  └──────────────┘
```

**Rules:**

1. Construction does **not** start an episode; first `reset` does.
2. `seed` is applied on `reset`, not on `init` (optional convenience init may stash `defaultSeed` applied on first reset only — must be documented).
3. `close` transitions to `closed` from any state; further `step`/`reset`/`render` throw.
4. Wrappers participate in the same lifecycle; `TimeLimit` may force `truncated` while inner still `running`.

---

## 11. Step & Reset Semantics

### 11.1 `reset`

```swift
func reset(
    seed: UInt64? = nil,
    options: ResetOptions? = nil
) throws -> ResetResult<Observation>
```

**`ResetResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `observation` | `Observation` | Initial obs; in `observationSpace` |
| `info` | `Info` | Diagnostics; may be empty |

**Semantics:**

1. Ends any current episode without emitting a final transition (caller discarded mid-episode state).
2. If `seed != nil`, re-seed all env RNG streams derived from this seed (see §13).
3. If `seed == nil`, continue RNG sequence from previous episode (do not reseed unless caller passes a seed).
4. `options` carries env-specific knobs (e.g. curriculum level, fixed initial state for debugging). Typed per env via `ResetOptions` protocol or `[String: ResetOptionValue]` with documented keys.
5. Does **not** return reward, terminated, or truncated.

### 11.2 `step`

```swift
func step(_ action: Action) throws -> StepResult<Observation>
```

**`StepResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `observation` | `Observation` | Successor observation `s'` |
| `reward` | `Reward` (default `Float`/`Double`) | Scalar reward for this transition |
| `terminated` | `Bool` | MDP terminal state reached |
| `truncated` | `Bool` | External / time / bound cutoff |
| `info` | `Info` | Extra data; may include final obs helpers in vector settings |

**Semantics:**

1. Requires env in `running` state (post-reset, not post-terminal without reset).
2. Applies `action`, advances dynamics one step.
3. Exactly one of these episode-end patterns:
   - both flags `false` → episode continues
   - `terminated == true` (truncation usually false)
   - `truncated == true` (termination usually false)
   - both `true` is **discouraged** but if it occurs, treat as terminal for bootstrapping purposes and document (prefer not to emit)
4. Reward is for the transition `(s, a, s')` just taken.
5. Observation is successor state `s'` even when the episode ended; vector autoreset may replace the returned observation with the next episode’s first obs — must set info keys accordingly (§16).

### 11.3 Reward type

**Locked:** scalar reward is always `Float` in single-env `StepResult` (see §7.3.1). Vector envs expose `MLXArray` rewards with shape `[numEnvs]` and dtype `.float32`. No `associatedtype Reward` in core protocols.

### 11.4 Typical agent loop (normative example)

```swift
var result = try env.reset(seed: 42)
var obs = result.observation

for _ in 0..<maxSteps {
    let action = policy(obs)  // or env.actionSpace.sample(using: &rng)
    let step = try env.step(action)
    obs = step.observation
    buffer.store(obs: obs, reward: step.reward, …)

    if step.terminated || step.truncated {
        // bootstrap: use terminated flag for value target (0 if true, V(s') if truncated-only)
        result = try env.reset()  // optional new seed
        obs = result.observation
    }
}
try env.close()
```

---

## 12. Termination vs Truncation

This distinction is **critical** for correct TD / advantage targets and is a first-class goal.

| Flag | Meaning | Typical causes | Bootstrap value at end? |
|------|---------|----------------|-------------------------|
| `terminated` | True absorbing / goal / failure state of the **task MDP** | Goal reached, fell over, game over, illegal terminal | Usually **no** (`V = 0`) |
| `truncated` | Episode stopped by **wrapper / limit / external constraint** | `TimeLimit`, wall-clock, safety stop, out-of-bounds cut | Usually **yes** (`V(s')`) |

**Implementer rules:**

1. Env dynamics set `terminated` only for true task terminals.
2. `TimeLimit` wrapper sets `truncated = true` when step count hits `maxEpisodeSteps`, and must **not** set `terminated` for that reason alone.
3. Algorithms in downstream packages should receive **both** flags; rlx-swift may provide a small pure helper (optional in `RLXCore`):

   ```swift
   func shouldBootstrap(terminated: Bool, truncated: Bool) -> Bool
   // true iff truncated && !terminated  (common convention; document alternatives)
   ```

4. Logging / stats: count completed episodes on `terminated || truncated`; label success via info or reward threshold separately.

---

## 13. Seeding & Determinism

### 13.1 Goals

- Same `(env_id, seed, actions sequence, platform, MLX backend config)` → same trajectories for **deterministic** envs (`EnvSpec.nondeterministic == false`).
- Stochastic envs: same seed ⇒ same stochastic outcomes (RNG draws identical).

### 13.2 Seed application

| Mechanism | Usage |
|-----------|--------|
| `reset(seed:)` | Primary; reseeds env PRNG tree |
| `options` fixed init state | Debugging / unit tests without full RNG |
| Space sampling | Pass explicit Swift RNG or MLX key; do not rely on env seed unless using shared `SeededSampler` |

### 13.3 PRNG tree (recommended implementation pattern)

On `reset(seed: S)`:

1. Create root MLX key `k0 = MLXRandom.key(S)`.
2. Split into streams: `env_dynamics`, `obs_noise`, `action_noise` (if any), etc.
3. Store stream keys on the env instance; each random op uses `split` to advance.
4. Do **not** call process-global `MLXRandom.seed` inside env code (avoids cross-talk between envs/tests).

Swift `RandomNumberGenerator` path: seed a deterministic RNG (e.g. `SplitMix64` / documented algorithm) stored on env for CPU-only envs.

### 13.4 Vector env seeding

- `reset(seed: base)` derives child seeds `hash(base, i)` or `split(key, numEnvs)` for sub-env `i`.
- Passing `seed: nil` advances each sub-env independently from its prior state.
- Optional `seed: [UInt64]` with length `numEnvs` for full control.

### 13.5 Limits of determinism (document in README)

- GPU / Metal floating point reductions may differ from CPU.
- Multithreaded async vector envs may reorder completions unless barriers are used; prefer deterministic mode for tests (`AsyncVectorEnv` serial fallback).
- External data (network, files) breaks purity — mark `nondeterministic`.

---

## 14. Info, Specs & Metadata

### 14.1 `Info`

Type-erased but structured bag for auxiliary outputs:

```swift
public struct Info: Equatable, Sendable {
    public subscript(key: String) -> InfoValue? { get set }
    // InfoValue: bool, int, double, string, MLXArray (careful with Sendable), nested Info
}
```

**Info key conventions (rlx-swift):**

Prefer **wrapper- or feature-scoped names** so stacked wrappers do not overwrite each other. Ship compile-time constants via `InfoKeys` to avoid typos.

| Key / pattern | Set by | Meaning |
|---------------|--------|---------|
| `TimeLimit.truncated` | `TimeLimit` wrapper | Optional diagnostic that truncation was due to the step limit (primary signal remains `truncated` on `StepResult`) |
| `final_observation` | Vector env autoreset | Terminal observation for a slot when the returned `observation` is already the next episode’s first obs |
| `final_info` | Vector env autoreset | Info from the terminal transition in the same situation |
| `episode` (nested `Info`) | `RecordEpisodeStatistics` | Completed episode metrics: `r` (return), `l` (length), optional `t` (elapsed time) |
| `rlx.*` | Future rlx-only extensions | Only for keys without an established name in this document |

**Environment-specific keys:** use env-scoped names, e.g. `cartpole.x_threshold_crossed`, or a single nested `Info` under the env id. Avoid unprefixed generic names (`score`, `done`) that collide across wrappers.

### 14.2 `EnvSpec` & instance config

Separate **kind metadata** (`EnvSpec`) from **instance configuration** (`CartPoleConfig`, render mode, dt, etc.). Factory:

```swift
registry.make("CartPole-v1", config: CartPoleConfig(...), renderMode: .none)
```

---

## 15. Wrappers & Transforms

### 15.1 Wrapper protocol requirements

1. Store `inner: InnerEnv`.
2. Default implementations forward `reset`/`step`/`close`/`spaces`/`render`.
3. Override only what changes; call `inner` for the rest.
4. Preserve or intentionally transform spaces; if obs/action types change, wrapper is a new `Environment` with new associated types (may require type erasure at stack edge).

### 15.2 v1 wrapper set

| Wrapper | Behaviour |
|---------|-----------|
| `TimeLimit` | Count steps since reset; on limit, `truncated = true` |
| `OrderEnforcing` | Throw if `step` before `reset` or `step` after end without reset |
| `PassiveEnvChecker` / `ActiveEnvChecker` | Validate spaces, dtypes, finite rewards (dev only) |
| `ClipAction` | Clip box actions to space bounds before inner step |
| `RescaleAction` | Linear map from policy space to env space |
| `TransformObservation` | Map `obs -> obs'` + new observation space |
| `TransformReward` | Map reward (e.g. scale, clip, sign) |
| `RecordEpisodeStatistics` | On episode end, push return/length into `info` |
| `Autoreset` (single-env, optional) | Auto `reset` on end; rare — prefer vector-level autoreset |
| `FrameStack` (optional v1.1) | Stack last k obs along new axis |

### 15.3 Composition

```swift
let env = RecordEpisodeStatistics(
    TimeLimit(
        OrderEnforcing(CartPole()),
        maxEpisodeSteps: 500
    )
)
```

Or builder DSL (optional ergonomic layer, not required in core):

```swift
CartPole()
    .wrapped(OrderEnforcing.init)
    .wrapped { TimeLimit($0, maxEpisodeSteps: 500) }
```

### 15.4 Transform purity

Transforms should be deterministic functions of inputs (plus documented running statistics for normalize wrappers). Running-mean wrappers must expose `updateStatistics: Bool` and serialization of stats for eval mode.

---

## 16. Vectorized & Parallel Environments

### 16.1 Why vectorize

Throughput for on-policy/off-policy collection: one policy forward over batched obs `[N, …]`, one `step` over batched actions.

### 16.2 `SyncVectorEnv`

- Owns `N` copies of the same env kind (separate instances).
- `step` loops sequentially (or internally parallel with serial result assembly — document).
- Simplest semantics; default for tests and small `N`.

### 16.3 `AsyncVectorEnv`

- Steps envs on worker threads / tasks (`Swift` structured concurrency).
- `step` awaits all workers; returns batched results in **fixed env index order**.
- Cancellation: `close` cancels outstanding tasks.
- Deterministic testing mode: `maxConcurrency: 1`.

### 16.4 Batch layout (MLX-first)

| Quantity | Shape |
|----------|-------|
| observations | `[numEnvs, *obsShape]` |
| actions | `[numEnvs, *actShape]` or `[numEnvs]` for discrete |
| rewards | `[numEnvs]` |
| terminated | `[numEnvs]` as `MLXArray` bool or `[Bool]` |
| truncated | same |

Provide helpers: `stack(_ values: [MLXArray])`, `unstack`, mask builders `episodeStillActive = !(terminated || truncated)` — careful with autoreset.

### 16.5 Autoreset modes

Vector autoreset timing must be **explicit and version-stable**. Supported modes:

| Mode | Behaviour |
|------|-----------|
| `.disabled` | Caller resets ended sub-envs explicitly (advanced) |
| `.nextStep` | On step *after* episode end, that slot returns first transition of new episode; previous step carried terminal transition with final obs in info |
| `.sameStep` | Within the ending step, automatically reset and return **new** episode’s first obs in `observation`, stash terminal obs in `info["final_observation"]` |

**Default for v1:** `.sameStep` on `SyncVectorEnv` (convenient for batched training: live observation is always “current episode”, terminal transition recoverable from info). Offer `.nextStep` for pipelines that want the terminal transition to stand alone before reset. Pin behaviour in tests; do not silently change defaults across minor versions.

**Requirement:** whichever mode is chosen, it is set at vector env construction and immutable; `info` keys always allow reconstruction of terminal transition.

### 16.6 Partial failure

If one sub-env throws, vector `step` fails entirely (v1). Future: `Result` per lane. Document that envs should not throw for normal terminals.

---

## 17. Rendering

### 17.1 `RenderMode`

Set at **construction** / `make`, not per `render()` call:

| Mode | Behaviour |
|------|-----------|
| `.none` | No render support required; `render()` returns nil or throws `notSupported` |
| `.human` | Side-effect display (window/console); return nil |
| `.rgbArray` | Return `MLXArray` or `ImageBuffer` HxWxC |
| `.ansi` | Return `String` frame (toy text envs) |

### 17.2 `Renderable` protocol

```swift
func render() throws -> RenderFrame?
```

`RenderFrame` enum: `.rgb(MLXArray)`, `.ansi(String)`, `.humanDisplayed`.

Rendering must **not** affect dynamics (except documented debug overlays).

---

## 18. Registry & Factory

### 18.1 Registration

```swift
EnvironmentRegistry.shared.register(
    id: "CartPole-v1",
    spec: EnvSpec(...),
    factory: { config, renderMode in
        AnyEnvironment(CartPole(config: config as! CartPoleConfig, renderMode: renderMode))
    }
)
```

Modules in `RLXEnvs` call `registerAllClassicControl()` from a public `registerDefaults()`.

### 18.2 `make`

```swift
try EnvironmentRegistry.shared.make(
    "CartPole-v1",
    config: nil,              // use defaults
    renderMode: .none,
    wrappers: [.timeLimit, .orderEnforcing]  // optional sugar
)
```

Unknown id → `RegistryError.unknownID`.  
Invalid config type → `RegistryError.invalidConfig`.

### 18.3 Discoverability

- `registry.ids: [String]`
- `registry.spec(for:)`
- Debug print / list helper for registered ids and specs.

---

## 19. Algorithm-Agnostic Integration Surface

Core must not implement PPO/DQN/etc., but should define **stable hooks** algorithms need.

### 19.1 Minimal policy interaction (documentation-level, optional thin protocols in core)

Not mandatory to ship in v1, but design reserves names:

| Protocol | Purpose |
|----------|---------|
| `Policy` | `func action(observation:) throws -> Action` (eval) |
| `StochasticPolicy` | returns action + logprob (+ entropy) as `MLXArray`s |
| `ValueFunction` | `V(s)` or `Q(s,a)` tensor interface |

These may live in `rlx-swift-algorithms` instead; mention here so env APIs (spaces, batch shapes) stay compatible.

### 19.2 Rollout / transition schema — **out of `rlx-swift` v1 core**

`rlx-swift` defines **how to interact with environments**. Rollout buffers, replay buffers, GAE, and advantage normalization depend on policy outputs (`log_prob`, `value`), optimizers, and on-policy vs off-policy semantics — they belong in **algorithm / training** packages that already depend on `MLXNN` and optimizers.

**v1 decision:** do **not** ship `RLXData` / `RolloutBuffer` / `ReplayBuffer` inside `rlx-swift`. Document a **recommended transition shape** for algorithm authors only:

```text
Transition(obs, action, reward, nextObs, terminated, truncated, info)
// Algorithm package adds: logProb, value, advantage, return, … as needed
```

Optional later: thin `rlx-swift-data` or buffers inside `rlx-swift-algorithms` — never a hard dependency of `RLXCore`.

### 19.3 What algorithms may assume

1. Step/reset contracts in §11–12 (`terminated` / `truncated` split).
2. Ability to batch via `VectorEnvironment` or manual stacking of `MLXArray`.
3. Seed control via `reset(seed:)`.
4. No hidden env stepping inside `render`/`close`.

### 19.4 What algorithms must not assume

1. Autoreset on single env (unless a wrapper explicitly documents it).
2. Process-global MLX seed mutation by env code.
3. A single combined `done` flag — always branch on `terminated` and `truncated` separately when bootstrapping.
4. Thread-safe sharing of one env instance across tasks without external sync.

---

## 20. Error Model & Validation

### 20.1 Error types

```swift
public enum EnvironmentError: Error {
    case notReset
    case episodeEnded  // step called without reset after terminal
    case closed
    case invalidAction(String)
    case invalidObservation(String)
    case renderNotSupported
    case configuration(String)
    case underlying(Error)
}

public enum SpaceError: Error { … }
public enum RegistryError: Error { … }
public enum VectorEnvironmentError: Error { … }
```

### 20.2 Validation layers

| Layer | When | Cost |
|-------|------|------|
| Debug asserts | `-assertions` / DEBUG | cheap |
| `OrderEnforcing` | opt-in wrapper | cheap |
| `PassiveEnvChecker` | opt-in / test | medium (contains checks each step) |
| `checkEnvironment(_:)` | unit/integration tests | multi-episode exercise |

### 20.3 `checkEnvironment` (testing utility)

Exercises:

1. Spaces sample + contains.
2. Reset produces valid obs.
3. Random policy steps without crash for N episodes.
4. After terminal, step without reset throws (if order enforcing installed).
5. Determinism: two envs, same seed, same action sequence → equal obs/rewards (if deterministic spec).
6. Close idempotency.

---

## 21. Concurrency & Thread Safety

| Object | Thread safety |
|--------|---------------|
| Single `Environment` instance | **Not** thread-safe; one task/thread at a time |
| `Space` (immutable config) | Safe to share read-only across threads |
| `EnvironmentRegistry` | Safe for concurrent `make` if registration completes before multi-threaded use; mutations serialized |
| `SyncVectorEnv` | External: treat as single-threaded unless documented internal locking |
| `AsyncVectorEnv` | Internal workers; public API `async` methods; `actor` or equivalent isolation recommended |

### 21.1 Single-env API: synchronous only (locked)

Keep the core interaction loop simple and MLX-friendly: most envs step in-process with tensor ops that are already async under the hood via MLX’s scheduler. Parallelism belongs at the **vector / driver** layer.

**v1 contract:**

| API surface | Style |
|-------------|--------|
| `Environment.reset` / `step` / `close` / `render` | **Synchronous** `throws` only |
| `SyncVectorEnv` | Synchronous batched `throws` |
| `AsyncVectorEnv` | **`async throws`** (optional sync façade that drives the actor for simple scripts) |

Do **not** add parallel `async` methods on `Environment` in v1 (avoids dual API and `Sendable` burden on every custom env). Envs that block on I/O still implement sync `step` and are driven from `AsyncVectorEnv` workers or user-level tasks.

MLX arrays: follow mlx-swift concurrency guidance; do not mutate shared `MLXArray` buffers from multiple threads.

---

## 22. Platform Strategy

| Platform | Priority | CI tier | Notes |
|----------|----------|---------|-------|
| macOS (Apple silicon) | P0 | **Tier-1** (build + test every PR) | Primary dev/train; Metal via MLX |
| iOS / iPadOS | P1 | **Tier-2** (compile smoke / optional simulator; not full test matrix in v1) | On-device envs + lightweight policies; watch memory |
| tvOS / visionOS | P2 | Tier-2 compile-only if package declares support | Inherit mlx-swift platform floor |
| Linux (mlx-swift CPU/CUDA where available) | P2 | Tier-2 when mlx-swift pin supports | Server collection workers |

**Platform declarations in `Package.swift`:** mirror mlx-swift’s supported Apple platforms at the dependency’s minimum OS versions (currently mlx-swift targets macOS 14+, iOS 17+, tvOS 17+, visionOS 1+ — **track the pinned mlx-swift release**, do not invent stricter/looser OS floors without cause).

**v1 CI industry norm for Swift libs:** one fully tested host (macOS) + optional cross-compile checks. Full iOS simulator RL tests are expensive and rarely gate P0; document iOS as supported compile target, promote to tier-1 tests only when maintainers commit device/simulator capacity.

Apple-first optimizations:

- Prefer in-process vector envs over multi-process (defer subprocess/remote workers to a later phase).
- Keep observations as `MLXArray` on GPU when policy is on GPU (minimize D2H copies).
- Optional Instruments signposts in examples, not core.

---

## 23. Testing Strategy

### 23.1 Unit tests (`RLXCoreTests`)

- Space sample/contains bounds.
- StepResult/ResetResult Codable/equality if applicable.
- PRNG split determinism.
- Registry register/make/unknown id.
- Error paths: not reset, closed, invalid action (with checker).

### 23.2 Contract tests (`RLXTesting` + env tests)

- Each shipped env passes `checkEnvironment`.
- Termination vs truncation: `TimeLimit` truncates without terminating CartPole artificially.
- Wrapper stack preserves inner unwrapped identity.

### 23.3 Vector tests

- Batch shapes.
- Autoreset info keys present on episode boundaries.
- Seeding: env i trajectory depends on `baseSeed` and `i` only.

### 23.4 Snapshot / golden trajectories

- Store seeded CartPole trajectories (obs hash + reward sum) for regression.
- Recompute only when dynamics intentionally change (version bump env id to `CartPole-v2`).

### 23.5 CI matrix

- macOS SwiftPM debug/release.
- Linux if supported by dependency pins.
- Optional iOS simulator build of core (no Metal-heavy tests).

---

## 24. Reference Environments (Bootstrap Set)

Ship a small set to validate the API (in `RLXEnvs`), not to compete with specialized simulators.

### 24.1 Classic control benchmarks (tensor-native implementations)

| ID | Obs space | Act space | Notes |
|----|-----------|-----------|-------|
| `CartPole-v1` | `Box(4,)` | `Discrete(2)` | Standard dynamics; max 500 steps via wrapper/spec |
| `MountainCar-v0` | `Box(2,)` | `Discrete(3)` | Sparse reward |
| `Pendulum-v1` | `Box(3,)` | `Box(1,)` continuous | Classic continuous control |
| `Acrobot-v1` | `Box(6,)` | `Discrete(3)` | Optional v1 |

Dynamics should match publicly documented control-theory / benchmark equations for each reference env. **v1 correctness bar:** equation-level unit tests, qualitative invariants (e.g. CartPole angle limits, reward signs), optional soft trajectory snapshots with tolerances on a fixed platform/seed. Bump env version id (`CartPole-v2`) if dynamics intentionally diverge from a prior rlx-swift release. Do not treat implementations outside rlx-swift as a required oracle.

### 24.2 Debugging envs

| ID | Purpose |
|----|---------|
| `IdentityEnv-v0` | Obs = action; tests wiring |
| `DummyEnv-v0` | Fixed length episodes; deterministic rewards |
| `ErrorEnv-v0` | Throws on command via options; tests error paths |

### 24.3 Out of scope for bootstrap

Atari, MuJoCo, DM Control, Unity — future adapter packages.

---

## 25. Phased Implementation Roadmap

| Phase | Deliverable | Exit criteria |
|-------|-------------|---------------|
| **P0 — Contracts** | `RLXCore`: protocols, `StepResult`/`ResetResult`, `Info`, errors, `Seed`/`PRNG` helpers, type erasure stubs | Compiles; unit tests for types |
| **P1 — Spaces** | `Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`, sample/contains, flatten helpers | Space tests green |
| **P2 — Minimal env + checker** | `DummyEnv`, `OrderEnforcing`, `checkEnvironment` | Contract tests pass |
| **P3 — Wrappers** | `TimeLimit`, stats, transform/clip/rescale | Wrapper tests + truncation semantics |
| **P4 — Registry** | Register/make/spec listing | Integration: make CartPole when ready |
| **P5 — Classic envs** | CartPole + one continuous (Pendulum) | Golden trajectory tests |
| **P6 — Vector** | `SyncVectorEnv`, autoreset mode, batch helpers | Vector tests; multi-env random rollout |
| **P7 — Async vector** | `AsyncVectorEnv` actor | Concurrency tests |
| **P8 — Polish** | Docs, examples random agent, DocC, performance notes | External contributor can implement custom env from docs alone |

Algorithms remain a **separate phase/repo** after P6+.

---

## 26. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Library identity | **RL infrastructure on mlx-swift** | Swift/`MLXArray` contracts; satellite package like `mlx-swift-lm` |
| Core dependency | `MLX` only in `RLXCore` | Algorithm-agnostic env layer; policies/optimizers elsewhere |
| Env reference semantics | Class-bound protocol (`AnyObject`) | Mutable episode state; wrapper identity |
| Spaces | Protocol + struct implementations | Value semantics for specs; tensor metadata via `MLXArray` / `DType` |
| Default tensor type | `MLXArray` for box-like data | First-class MLX training interchange |
| Dual typed / type-erased API | Generics + **manual** `AnyEnvironment` eraser | Swift `associatedtype` erasure; registry/`make` ergonomics |
| Global RNG in envs | Forbidden; explicit MLX keys / Swift RNG | Reproducibility; test isolation |
| Time limit handling | Wrapper sets `truncated`; optional `TimeLimit.truncated` in `info` | Task MDP stays clean; diagnostics available |
| Vector autoreset default | Configurable; default `.sameStep`; offer `.nextStep` | Explicit modes; stable across versions |
| Multi-process vec env | Defer | In-process + MLX batching first on Apple silicon |
| Algorithms in-repo | No (`rlx-swift-algorithms` later) | Clear module boundary; mirrors mlx-swift vs mlx-swift-lm split |
| Rollout buffers in v1 | **No** | Buffers need policies/optimizers; algorithm package concern |
| Render mode timing | Construction / `make` only | Avoid mid-episode mode flips |
| Error style | `throws` + typed errors | Idiomatic Swift |
| Package location | **Standalone** `rlx-swift` repo depending on mlx-swift | Independent versioning; satellite package pattern |
| License | **MIT** (match mlx-swift) | Ecosystem consistency |
| Toolchain | Swift tools-version **aligned to pinned mlx-swift** (6.3+ today) | Must meet dependency requirement |
| Platform minimums | **Inherit pinned mlx-swift** | Single source of truth for OS floors |
| Reward scalar | **`Float` fixed**; vector `MLXArray` f32 | Aligns with MLX float32 training; avoids generic explosion |
| Single-env concurrency | **Sync only**; async on `AsyncVectorEnv` | Simple core API; parallelism at vector/driver layer |
| Reference env numerics | Equation + invariant tests | Correct dynamics without external-runtime oracles |
| Info key names | Documented constants (`final_observation`, `TimeLimit.truncated`, `episode`, `rlx.*` for extensions) | Stable, collision-resistant diagnostics |
| v1 CI tier-1 | **macOS** build+test; other platforms compile smoke optional | Practical Swift package CI |

---

## 27. Resolved Questions (formerly open)

All items below are **locked for v1** unless a future RFC explicitly revisits them. Decisions follow **mlx-swift ecosystem norms**, Swift language practice, and clean RL layering (env core vs algorithms/training).

### 27.1 Package identity — **standalone repository**

| Option | Verdict |
|--------|---------|
| Standalone `rlx-swift` SwiftPM repo depending on `mlx-swift` | **Chosen** |
| Subfolder inside mlx-swift | Rejected — couples unrelated release cadences; mlx-swift is arrays/runtime, not RL |
| Org monorepo with multiple packages | Acceptable later; still **separate package identity** and version tags |

**Anchor:** `mlx-swift-lm` and `mlx-swift-examples` are separate repos consuming `mlx-swift`, not in-tree modules. rlx-swift is the same kind of satellite: RL on MLX, not a mlx-swift submodule.

### 27.2 Minimum Swift / mlx-swift versions — **pin dependency; inherit tools-version**

| Rule | Detail |
|------|--------|
| `mlx-swift` dependency | Pin to an **exact released tag** (or narrow range) in `Package.swift`; document in README |
| `swift-tools-version` | Set to **≥** what the pinned mlx-swift requires (track the pin — e.g. **6.3** on current mlx-swift main, not unpinned “latest”) |
| Swift language mode | Swift 6 preferred when the dependency allows |
| Bump policy | Breaking mlx-swift upgrades may bump rlx-swift major/minor; CI tests against the pin only |

### 27.3 License — **MIT**

| Fact | Implication |
|------|-------------|
| mlx-swift ships under **MIT** | rlx-swift uses **MIT** for ecosystem consistency |
| GPL/copyleft adapters | Keep out of core targets |

### 27.4 `AnyEnvironment` storage — **manual type-eraser class (primary)**

| Approach | Role |
|----------|------|
| `final class AnyEnvironment` (boxed inner env / closures) | **Primary public type** for registry, `make`, heterogeneous collections |
| `any Environment` existential | Internal or generic helpers only when sufficient |
| Open `BaseEnv` class hierarchy as sole model | Rejected — protocols + wrappers compose better for MLX-backed envs |

**Anchor:** Swift uses explicit erasers (`AnySequence`, `AnyPublisher`) when protocols have `associatedtype`. Registry/`make` needs a stable concrete type without forcing every caller into generics.

**Requirements on `AnyEnvironment`:**

1. Forwards `reset` / `step` / `close` / spaces (spaces as `AnySpace`).
2. Preserves `unwrapped` chain for wrappers.
3. Surfaces failures as `EnvironmentError` / `Error`, not erased away silently.

### 27.5 Reward type — **`Float` fixed in core**

See §7.3.1 and §11.3. **Rejected:** `associatedtype Reward` on `Environment`. **Rejected:** default `Double` at the public step boundary (misaligned with MLX float32 training). Internal physics may use higher precision; cast to `Float` when returning `StepResult`.

### 27.6 Async API on single env — **sync only; async at vector/driver**

See §21.1. **Rejected for v1:** `async` on every `Environment`. **Accepted:** `AsyncVectorEnv` with `async throws`; optional sync façade for scripts.

### 27.7 Reference env correctness — **equations & invariants, not external oracles**

| Tier | Requirement |
|------|-------------|
| Must | Documented space shapes/dtypes; correct `terminated` / `truncated` use; sensible reward structure |
| Must | Unit tests against stated dynamics equations (with tolerances) |
| Should | Invariant tests (e.g. CartPole ends past angle/position thresholds) |
| Optional | Seeded trajectory snapshots on macOS/Metal with explicit tolerances |
| Not required | Bit-identical trajectories vs any non-rlx implementation |

Version env ids (`CartPole-v2`) when rlx-swift dynamics intentionally change.

### 27.8 Info key names — **documented constants; `rlx.` for extensions**

See §14.1. Fixed names for vector/stats/time-limit diagnostics; `rlx.*` only for keys introduced uniquely by this project. Goal: stable logging/hooks inside the MLX/Swift RL stack, not compatibility with foreign frameworks.

### 27.9 `RLXData` / rollout buffers in v1 — **no**

See §19.2. Env core = interaction; algorithm package = buffers + GAE + optimizer steps on `MLXNN` modules.

### 27.10 iOS as tier-1 CI — **no for v1 tests; yes as platform declaration**

| Layer | v1 policy |
|-------|-----------|
| `Package.swift` platforms | Align with pinned mlx-swift (iOS/tvOS/visionOS as applicable) |
| PR CI tier-1 | **macOS** `swift test` (or xcodebuild macOS) every PR |
| PR CI tier-2 | Optional iOS simulator **build**; full env suites not required initially |
| Promote iOS tests | When maintainers commit simulator/device CI capacity |

---

## 28. PR Plan

Incremental, reviewable PRs for `rlx-swift` implementation. Each PR should merge independently with tests.

### PR-01 — Repository scaffold
- **Title:** `chore: scaffold rlx-swift package and RLXCore target`
- **Affects:** `Package.swift` (MIT LICENSE, platforms/tools-version inherited from **pinned** mlx-swift tag), empty `Sources/RLXCore`, `Tests/RLXCoreTests`, README stub, this `design.md` copy
- **Depends on:** —
- **Description:** Standalone repo; SwiftPM dep on mlx-swift pin; tier-1 macOS CI (`swift test`); optional tier-2 iOS compile job; no behaviour yet.

### PR-02 — Core result types & errors
- **Title:** `feat(core): StepResult, ResetResult, Info, EnvironmentError`
- **Affects:** `Sources/RLXCore/StepResult.swift`, `ResetResult.swift`, `Info.swift`, `Errors.swift`, `RenderMode.swift`
- **Depends on:** PR-01
- **Description:** Value types + unit tests for info key access and error equality.

### PR-03 — Seed & PRNG utilities
- **Title:** `feat(core): Seed, PRNG key tree helpers on MLX`
- **Affects:** `Sources/RLXCore/Seed.swift`, `PRNG.swift`
- **Depends on:** PR-02
- **Description:** Deterministic split helpers; forbid global seed in API docs; tests for key reproducibility.

### PR-04 — Space protocol + Discrete & Box
- **Title:** `feat(spaces): Space protocol, DiscreteSpace, BoxSpace`
- **Affects:** `Sources/RLXCore/Space.swift`, `Spaces/DiscreteSpace.swift`, `Spaces/BoxSpace.swift`
- **Depends on:** PR-03
- **Description:** `sample`/`contains` for Swift RNG + MLX key; shape/dtype metadata.

### PR-05 — MultiDiscrete, MultiBinary, Dict/Tuple spaces
- **Title:** `feat(spaces): composite and multi spaces + flatten helpers`
- **Affects:** `Sources/RLXCore/Spaces/*`, `SpaceFlatten.swift`
- **Depends on:** PR-04
- **Description:** Composite membership/sample; deterministic key order for dict flatten.

### PR-06 — Environment protocol + type erasure
- **Title:** `feat(core): Environment protocol and AnyEnvironment`
- **Affects:** `Sources/RLXCore/Environment.swift`, `TypeErasure.swift` (`AnyEnvironment` / `AnySpace` eraser classes), `EnvSpec.swift`
- **Depends on:** PR-05
- **Description:** Normative sync protocol methods; manual type erasers for registry; spec struct.

### PR-07 — DummyEnv + OrderEnforcing + checkEnvironment
- **Title:** `feat(testing): DummyEnv, OrderEnforcing, checkEnvironment`
- **Affects:** `Sources/RLXEnvs/DummyEnv.swift` or under Testing, `Sources/RLXWrappers/OrderEnforcing.swift`, `Sources/RLXTesting/CheckEnvironment.swift`
- **Depends on:** PR-06
- **Description:** First runnable env; contract test harness; lifecycle enforcement.

### PR-08 — TimeLimit & episode statistics wrappers
- **Title:** `feat(wrappers): TimeLimit and RecordEpisodeStatistics`
- **Affects:** `Sources/RLXWrappers/TimeLimit.swift`, `RecordEpisodeStatistics.swift`, `InfoKeys.swift`
- **Depends on:** PR-07
- **Description:** Truncation semantics; `InfoKeys` for `TimeLimit.truncated`, `episode` metrics (`r`, `l`).

### PR-09 — Action/observation/reward transform wrappers
- **Title:** `feat(wrappers): ClipAction, RescaleAction, TransformObservation/Reward`
- **Affects:** `Sources/RLXWrappers/Transforms/*.swift`
- **Depends on:** PR-08
- **Description:** Space remapping tests; pure transform properties.

### PR-10 — Environment registry & make
- **Title:** `feat(core): EnvironmentRegistry register/make/list`
- **Affects:** `Sources/RLXCore/Registry.swift`
- **Depends on:** PR-06
- **Description:** Duplicate-id policy; factory config typing tests.

### PR-11 — CartPole-v1 reference env
- **Title:** `feat(envs): CartPole-v1 tensor-native implementation`
- **Affects:** `Sources/RLXEnvs/ClassicControl/CartPole.swift`, registration
- **Depends on:** PR-10, PR-08
- **Description:** Dynamics + default wrapper stack in factory; golden trajectory test.

### PR-12 — Pendulum-v1 (continuous actions)
- **Title:** `feat(envs): Pendulum-v1 continuous control env`
- **Affects:** `Sources/RLXEnvs/ClassicControl/Pendulum.swift`
- **Depends on:** PR-11
- **Description:** Box action path validation end-to-end.

### PR-13 — SyncVectorEnv + autoreset
- **Title:** `feat(vector): SyncVectorEnv with configurable autoreset mode`
- **Affects:** `Sources/RLXVector/SyncVectorEnv.swift`, `AutoresetMode.swift`, batch helpers
- **Depends on:** PR-11
- **Description:** Batched step/reset; `final_observation` / `final_info`; default `.sameStep` + tested `.nextStep`; seed-per-env tests.

### PR-14 — AsyncVectorEnv
- **Title:** `feat(vector): AsyncVectorEnv using Swift concurrency`
- **Affects:** `Sources/RLXVector/AsyncVectorEnv.swift`
- **Depends on:** PR-13
- **Description:** Actor-isolated implementation; ordering guarantees; cancel on close.

### PR-15 — PassiveEnvChecker & docs/examples
- **Title:** `docs: DocC outlines, random agent example, design cross-links`
- **Affects:** `Sources/RLXWrappers/PassiveEnvChecker.swift`, `Examples/RandomAgentDemo`, DocC catalogs
- **Depends on:** PR-14
- **Description:** Contributor guide: implement a custom env in 50 lines; link to this design doc.

### PR-16 — (Future repo) Algorithms placeholder
- **Title:** `chore: note algorithms + buffers live in rlx-swift-algorithms`
- **Affects:** README only in `rlx-swift`
- **Depends on:** PR-15
- **Description:** No rollout/replay buffers in rlx-swift; documents env-vs-algorithms split for PPO/DQN/GAE work on MLXNN.

---

## Appendix A — Protocol Sketch (Normative Pseudocode)

Swift-like signatures for implementers. Names are normative; locked decisions in §26–§27 override earlier draft ambiguity.

```swift
// MARK: - Spaces

public protocol Space<Value>: Sendable {
    associatedtype Value
    var shape: [Int]? { get }
    var dtype: DType? { get }

    func contains(_ value: Value) -> Bool
    func sample(using rng: inout some RandomNumberGenerator) -> Value
    func sample(key: MLXArray) -> Value
}

// MARK: - Results

public struct ResetResult<Observation> {
    public var observation: Observation
    public var info: Info
}

public struct StepResult<Observation> {
    public var observation: Observation
    public var reward: Float
    public var terminated: Bool
    public var truncated: Bool
    public var info: Info
}

// MARK: - Environment

public protocol Environment: AnyObject {
    associatedtype Observation
    associatedtype Action
    associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation
    associatedtype ActionSpace: Space where ActionSpace.Value == Action

    var observationSpace: ObservationSpace { get }
    var actionSpace: ActionSpace { get }
    var spec: EnvSpec? { get }   // optional but recommended

    func reset(seed: UInt64?, options: ResetOptions?) throws -> ResetResult<Observation>
    func step(_ action: Action) throws -> StepResult<Observation>
    func close() throws
}

public protocol Renderable {
    func render() throws -> RenderFrame?
}

// MARK: - Wrapper

public protocol EnvironmentWrapper: Environment {
    associatedtype Inner: Environment
    var inner: Inner { get }
    var unwrapped: any AnyEnvironmentProtocol { get }
}

// MARK: - Vector

public enum AutoresetMode: Sendable {
    case disabled
    case nextStep
    case sameStep
}

public protocol VectorEnvironment: AnyObject {
    associatedtype Observation   // typically MLXArray
    associatedtype Action

    var numEnvs: Int { get }
    var autoresetMode: AutoresetMode { get }
    var singleObservationSpace: any AnySpaceProtocol { get }
    var singleActionSpace: any AnySpaceProtocol { get }

    func reset(seed: UInt64?, options: ResetOptions?) throws -> ResetResult<Observation>
    func step(actions: Action) throws -> StepResult<Observation>
    func close() throws
}

// MARK: - Registry

public protocol EnvironmentFactory: Sendable {
    func make(config: (any EnvConfig)?, renderMode: RenderMode) throws -> AnyEnvironment
}

public final class EnvironmentRegistry: @unchecked Sendable {
    public static let shared: EnvironmentRegistry
    public func register(id: String, spec: EnvSpec, factory: EnvironmentFactory) throws
    public func make(_ id: String, config: (any EnvConfig)?, renderMode: RenderMode) throws -> AnyEnvironment
    public var ids: [String] { get }
    public func spec(for id: String) -> EnvSpec?
}
```

---

## Appendix B — Glossary

| Term | Definition |
|------|------------|
| **Episode** | Trajectory from `reset` until `terminated \|\| truncated` |
| **Transition** | Single `(s, a, r, s', terminated, truncated)` step outcome |
| **Termination** | Task-defined absorbing / success / failure end |
| **Truncation** | External stop (time limit, etc.) |
| **Bootstrap** | Using `V(s')` as target when episode truncated but not terminated |
| **Vector env** | Batched interface over N env instances |
| **Autoreset** | Vector policy that starts next episode automatically in a slot |
| **Space** | Set descriptor for valid obs/actions + sampling |
| **Wrapper** | Composable env adapter preserving or transforming interface |
| **Type erasure** | `AnyEnvironment` hiding concrete `Observation`/`Action` types |
| **PRNG tree** | Split MLX keys / Swift RNG streams for isolated randomness |
| **EnvSpec** | Immutable metadata for an environment ID |
| **Info** | Side-channel diagnostics dictionary on reset/step |
| **Info key (standard)** | Documented rlx diagnostic key (`final_observation`, `TimeLimit.truncated`, `episode`, …) |
| **Info key (`rlx.`)** | Extensions unique to this library |
| **Type eraser** | Concrete `AnyEnvironment` / `AnySpace` class hiding associated types (Swift idiom) |
| **MLX-native** | APIs and data paths designed for `MLXArray` / MLX PRNG first |

---

## Document Control

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2026-06-25 | Initial design for rlx-swift; implementation not started |
| 1.1 | 2026-06-25 | Resolved §27; MIT license; locked reward/async/CI/buffer boundaries |
| 1.2 | 2026-06-25 | MLX-native RL identity; removed external-framework-centric framing and compatibility appendix |

**Maintenance:** Update this document in the same PR as any intentional API contract change. Implementation details that do not affect contracts belong in DocC / code comments, not here.

---

*End of design.md — single source of truth for rlx-swift architecture and initial protocols.*
