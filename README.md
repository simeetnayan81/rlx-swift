# rlx-swift

Reinforcement learning infrastructure for Swift, built on [mlx-swift](https://github.com/ml-explore/mlx-swift).

`rlx-swift` supplies the environment and data-collection substrate for MLX training code: MDP interaction (`reset` / `step`), observation/action spaces, wrappers, vectorized execution, and seeding — with `MLXArray` as the primary tensor type.

It is **not** an algorithms package. Policies, losses, and optimizers live in separate targets/packages (`MLXNN`, `MLXOptimizers`, future `rlx-swift-algorithms`).

> **Status:** Core result types (`StepResult`, `ResetResult`, `Info`, `EnvironmentError`, `RenderMode`). See [design.md](design.md) for architecture, API contracts, and the PR implementation plan.

## Requirements

| Item | Version / note |
|------|----------------|
| Swift | 6.0+ (`swift-tools-version: 6.0`; mlx-swift 0.31.4 pin is 5.12) |
| Platforms (declared) | macOS 14+, iOS 17+, tvOS 17+, visionOS 1+ |
| Linux (tier-2) | Supported for **CPU** mlx-swift builds only; see [Building on Linux](#building-on-linux) |
| mlx-swift | `0.31.x` (SwiftPM pin in `Package.swift`) |

**Tier-1 CI:** macOS `xcodebuild` build+test (Metal). **Tier-2:** Linux compile + `RLXCoreSmoke` (CPU, no Metal). Other Apple platforms are compile targets (optional).

## Quick start

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/<org>/rlx-swift", from: "0.1.0"),
],
// target dependency: .product(name: "RLXCore", package: "rlx-swift")
```

### macOS

```bash
swift build && swift run RLXCoreSmoke          # link smoke (no Metal runtime eval)
xcodebuild test -scheme rlx-swift-Package -destination 'platform=macOS'   # full tests (Metal)
# or: ./scripts/xcodebuild-test.sh
```

### Linux

See [Building on Linux](#building-on-linux) below, or run:

```bash
./scripts/linux-smoke.sh
```

## Building on Linux

Linux is **tier-2**: useful for server-side collection workers and CI without a Mac. mlx-swift uses its **CPU backend** (no Metal). Full XCTest suites that evaluate `MLXArray` may still need a Mac/Metal path; the supported Linux gate is **build + `RLXCoreSmoke`**.

### 1. Install Swift

Install Swift **6.0+** for your distro from [swift.org/install/linux](https://www.swift.org/install/linux/). Confirm:

```bash
swift --version   # 6.0 or newer
```

A C++ toolchain is required to compile mlx-swift’s `Cmlx` (usually comes with `build-essential` / `gcc-c++`).

### 2. Install BLAS / LAPACK (required)

On Linux, mlx-swift does **not** use Accelerate. The CPU backend includes `<cblas.h>` / LAPACK and links `openblas`, `blas`, `lapack`, and `gfortran`. Without the **dev** packages you get:

```text
fatal error: 'cblas.h' file not found
```

**Debian / Ubuntu** (and most containers):

```bash
sudo apt-get update -y
sudo apt-get install -y \
  build-essential \
  libblas-dev \
  liblapacke-dev \
  libopenblas-dev \
  gfortran
```

**Fedora / RHEL-style:**

```bash
sudo dnf install -y \
  gcc-c++ \
  make \
  blas-devel \
  lapack-devel \
  openblas-devel \
  gcc-gfortran
```

Sanity-check headers (paths vary by distro):

```bash
# one of these should exist
ls /usr/include/cblas.h \
   /usr/include/x86_64-linux-gnu/cblas.h \
   /usr/include/openblas/cblas.h 2>/dev/null
```

These packages match [mlx-swift’s Linux CI setup](https://github.com/ml-explore/mlx-swift/blob/main/.github/scripts/setup%2Bbuild-linux-container-cmake.sh).

### 3. Build

From the repo root:

```bash
git clone https://github.com/<org>/rlx-swift.git
cd rlx-swift

swift package resolve
swift build
```

First build compiles mlx-swift `Cmlx` from source and can take several minutes. On failure, wipe the cache and retry after installing deps:

```bash
rm -rf .build
swift build
```

Useful variants:

```bash
swift build -c release          # optimized
swift build --product RLXCore   # library only
swift build --product RLXCoreSmoke
```

### 4. Smoke test

`RLXCoreSmoke` is the Linux-friendly gate: it links `RLXCore` + `MLX` at **build** time and checks package identity at **run** time **without** calling `MLXArray` eval (avoids Metal/runtime resource issues on CLI toolchains).

```bash
swift run RLXCoreSmoke
# or, after a successful build:
.build/debug/RLXCoreSmoke
# release:
# swift run -c release RLXCoreSmoke
```

Expected output (exit code `0`):

```text
RLXCoreSmoke: all checks passed (rlx-swift 0.1.0, RLXCore+MLX linked at build time)
```

One-shot helper (install deps yourself first, or pass `--install-deps` on Debian/Ubuntu with sudo):

```bash
./scripts/linux-smoke.sh
# ./scripts/linux-smoke.sh --install-deps   # apt only
# ./scripts/linux-smoke.sh --release
```

### 4b. Linux via Docker (macOS or any Docker host)

Tier-2 Linux build + `RLXCoreSmoke` without a Linux VM. The repo root [`Dockerfile`](Dockerfile) is based on official `swift:6.0`, installs OpenBLAS/LAPACK, and runs [`scripts/linux-smoke.sh`](scripts/linux-smoke.sh).

```bash
# One-shot helper (builds image, bind-mounts \$PWD, runs smoke)
./scripts/linux-smoke-docker.sh
# ./scripts/linux-smoke-docker.sh --release
# ./scripts/linux-smoke-docker.sh --rebuild   # force image rebuild

# Or manually:
docker build -t rlx-swift-linux .
docker run --rm rlx-swift-linux                          # sources baked at build time
docker run --rm -v "$PWD":/workspace:ro rlx-swift-linux # live host tree (no rebuild per edit)
```

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac) or Docker Engine. First build compiles mlx-swift `Cmlx` for Linux and can take several minutes.

### 5. What Linux does *not* cover (yet)

| Check | Linux | Notes |
|-------|--------|--------|
| `swift build` (`RLXCore`, `RLXCoreSmoke`) | Yes | CPU backend |
| `swift run RLXCoreSmoke` | Yes | Link + identity only |
| `swift test` / XCTest with `MLXArray` eval | Not tier-1 | Prefer macOS + `xcodebuild` for Metal-backed runtime tests |
| Metal / GPU | No | CUDA is an mlx-swift Linux option for MLX itself; not wired as rlx tier-1 |
| iOS / visionOS products | No | Apple SDKs only |

For full tests on a Mac:

```bash
./scripts/xcodebuild-test.sh
```

## Package layout

| Target | Role | Status |
|--------|------|--------|
| `RLXCore` | Protocols, spaces, results, seed, errors, registry | Core result types (PR-02) |
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
