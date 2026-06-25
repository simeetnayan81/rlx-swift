# rlx-swift

Reinforcement learning infrastructure for Swift, built on [mlx-swift](https://github.com/ml-explore/mlx-swift).

`rlx-swift` supplies the environment and data-collection substrate for MLX training code: MDP interaction (`reset` / `step`), observation/action spaces, wrappers, vectorized execution, and seeding — with `MLXArray` as the primary tensor type.

It is **not** an algorithms package. Policies, losses, and optimizers live in separate targets/packages (`MLXNN`, `MLXOptimizers`, future `rlx-swift-algorithms`).

> **Status:** Early scaffold. See [design.md](design.md) for architecture, API contracts, and the PR implementation plan.

## Requirements

| Item | Version / note |
|------|----------------|
| Swift | 6.3+ (aligned with pinned mlx-swift) |
| Platforms | macOS 14+, iOS 17+, tvOS 17+, visionOS 1+ |
| mlx-swift | `0.31.x` (SwiftPM pin in `Package.swift`) |

**Tier-1 CI:** macOS `swift test`. Other Apple platforms are compile targets (tier-2 smoke optional).

## Quick start

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/<org>/rlx-swift", from: "0.1.0"),
],
// target dependency: .product(name: "RLXCore", package: "rlx-swift")
```

```bash
swift build
swift test          # requires full Xcode (not Command Line Tools only)
swift run RLXCoreSmoke   # scaffold smoke checks without XCTest runtime
```

## Package layout

| Target | Role | Status |
|--------|------|--------|
| `RLXCore` | Protocols, spaces, results, seed, errors, registry | Scaffold (PR-01) |
| `RLXWrappers` | TimeLimit, transforms, order enforcement | Planned |
| `RLXVector` | Sync / async vector envs | Planned |
| `RLXEnvs` | Reference envs (CartPole, Pendulum, …) | Planned |
| `RLXTesting` | `checkEnvironment`, contract helpers | Planned |

Full layout and contracts: [design.md](design.md) §6–§8.

## Design

The authoritative design document is [design.md](design.md):

- Goals, non-goals, and design principles
- `Environment` / `Space` / `StepResult` contracts
- mlx-swift integration rules (`RLXCore` depends on `MLX` only)
- Phased PR plan (§28)

## License

MIT — see [LICENSE](LICENSE). Matches the mlx-swift ecosystem.
