# Design: ALE (Atari) C++ adapter for rlx-swift

| Field | Value |
|-------|--------|
| Status | **Draft** — design and implementation plan only; no production code yet |
| Branch | `docs/ale-adapter-design` |
| Related | Repository [`design.md`](../design.md) §2 (out of scope), §24.3 (Atari = future adapter), §27.3 (GPL/copyleft adapters out of core) |
| Primary dependency (planned) | [Farama Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) (ALE) **C++** library |

This document defines **how** rlx-swift will integrate Atari via ALE’s C++ API, **where** the code lives, and a **phased implementation plan**. It does not implement the adapter.

---

## 1. Motivation

`rlx-swift` provides an MLX-native **environment substrate** (`Environment`, spaces, wrappers, vector envs). Classic control (CartPole, Pendulum, DummyEnv) validates that API.

**Atari via ALE** is the standard image-based discrete-control benchmark for deep RL. Integrating it:

- Proves the substrate under **high-dimensional observations** (`MLXArray` images).
- Unlocks future algorithms work (e.g. DQN) without stuffing simulators into core.
- Follows the design rule that Atari is an **adapter**, not a core env.

---

## 2. Goals and non-goals

### 2.1 Goals (adapter package)

1. Expose one or more types conforming to ``Environment`` that drive ALE C++ for a chosen ROM.
2. Observation as tensor-friendly data (`MLXArray`), actions as discrete `Int`.
3. Lifecycle aligned with rlx-swift: `reset` / `step` / `close`, `terminated` vs `truncated`, explicit seeding where ALE allows.
4. Optional registration of stable env ids for `EnvironmentRegistry.make`.
5. Documented build: how to obtain/link ALE, supply ROMs, run a minimal demo.
6. Work under existing ``SyncVectorEnv`` / ``AsyncVectorEnv`` as **N independent ALE instances** (phase 1).

### 2.2 Non-goals (this effort)

| Non-goal | Rationale |
|----------|-----------|
| Shipping ALE or ROMs inside `RLXCore` / default products | Keeps core MLX-only; license/ROM policy |
| Bit-identical trajectories vs Gymnasium/ALE-Python | design.md rejects cross-runtime bit-identity |
| Full Atari suite automation on day one | One game path first |
| Native ALE multi-env C++ vectorizer (phase 1) | Defer until N× single envs are proven |
| Algorithms (DQN, etc.) in this package | Belong in `rlx-swift-algorithms` or user code |
| SDL training path as required | Optional render; training uses array obs |
| iOS/App Store Atari product | Out of scope for research adapter v1 |

---

## 3. Package and repository layout

### 3.1 Decision: separate product, not core target

| Approach | Verdict |
|----------|---------|
| New optional product **inside** `rlx-swift` monorepo (`RLXALE`) | **Accepted for v1 of the adapter** — simpler iteration; still **optional** product (not linked by default consumers) |
| Separate git repo `rlx-swift-ale` only | Valid alternative later; monorepo first reduces version skew during spike |
| Add ALE to `RLXEnvs` | **Rejected** — pulls C++/ROMs into classic-control package |

**Planned monorepo layout (when implementation starts):**

```text
rlx-swift/
  Sources/RLXCore/ …          # unchanged; no ALE dependency
  Sources/RLXALE/             # Swift Environment + config + registration
  Sources/RLXALECXX/          # or cxx shim target: thin C API over ALE C++
  Examples/ALERandomAgent/    # optional executable (requires ALE + ROM)
  Tests/RLXALETests/          # gated / skipped without ALE+ROM
  docs/ale-adapter-design.md  # this document
  Package.swift               # optional products; core products unchanged
```

Consumers who only want CartPole never link `RLXALE`.

### 3.2 Dependency direction

```text
RLXALE  →  RLXCore, RLXWrappers (optional stack helpers)
        →  RLXALECXX / system ALE (C++)
RLXCore →  MLX only  (unchanged)
```

**Hard rule:** No ALE headers or link flags on `RLXCore`, `RLXWrappers`, `RLXEnvs` (classic), `RLXTesting`, or `RLXVector`.

---

## 4. Upstream: ALE C++

### 4.1 Library

- **Upstream:** Farama-Foundation Arcade Learning Environment.
- **API surface (conceptual):** `ALEInterface` (or current equivalent): load ROM, reset, act, screen/RAM buffers, legal actions, lives, game over, frameskip, etc.
- **Build:** CMake C++17; optional SDL for display (default **off** for CI/training).
- **Pinning:** Implementation PRs must pin a **specific ALE version/commit** and document it here and in adapter README.

### 4.2 Bridging strategy

**Chosen for implementation:** thin **C-compatible shim** (or minimal C++ `extern "C"` façade) + Swift import.

```text
Swift RLXALE (Environment)
        │
        ▼
C API (rlx_ale_*)  — stable, no Swift-visible STL
        │
        ▼
ALE C++ (ALEInterface / current types)
```

**Rejected for first cut:** pure Swift C++ interop over full ALE headers (high build/interop risk). Revisit only if shim maintenance is worse than interop pain.

### 4.3 Threading model

- **One ALE instance per Swift env instance.**
- Do **not** share one emulator across threads/tasks.
- Compatible with ``AsyncVectorEnv`` only as **one instance per slot**.

---

## 5. Environment contract mapping

### 5.1 Core types

| ALE concept | rlx-swift mapping |
|-------------|-------------------|
| Legal / minimal action set | ``DiscreteSpace`` with `n = actionCount`; action = `Int` index into legal set (default **minimal** action set) |
| Screen RGB or grayscale | ``BoxSpace`` + ``MLXArray`` observation (dtype/shape documented on env) |
| RAM (optional mode) | Separate config path: 128-byte RAM → `MLXArray` shape `[128]` or fixed `Box` |
| Step reward | ``StepResult/reward`` as `Float` |
| Game over | ``terminated = true`` (task end) |
| Time / step budget | Prefer ``TimeLimit`` wrapper (truncated), not buried only in C++ |
| Life loss | **Configurable policy** (see §5.3); not hard-coded without documentation |
| Soft/hard reset | ``reset(seed:options:)``; seed applied when ALE supports it |
| Destroy emulator | ``close()`` |

### 5.2 Proposed public Swift surface (sketch — not implemented)

```swift
// Product: RLXALE
public struct ALEConfig: Sendable {
    public var romPath: String
    public var obsType: ALEObservationType   // .rgb, .grayscale, .ram
    public var actionSet: ALEActionSet       // .minimal, .full
    public var frameskip: Int                // or fixed 4 with max-pool option later
    public var repeatActionProbability: Float // sticky actions; 0 = off
    public var fullActionSpace: Bool
    // lives policy, max pools, etc.
}

public enum ALEObservationType: String, Sendable { case rgb, grayscale, ram }

public final class ALEEnvironment: Environment {
    public typealias Observation = MLXArray
    public typealias Action = Int
    // observationSpace: BoxSpace
    // actionSpace: DiscreteSpace
    public init(config: ALEConfig) throws
    public func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<MLXArray>
    public func step(_ action: Int) throws -> StepResult<MLXArray>
    public func close() throws
}
```

Exact property names and defaults are fixed at implementation time and recorded in this doc’s revision history.

### 5.3 Lives policy

| Mode | Behaviour |
|------|-----------|
| `gameOverOnly` | `terminated` only on ALE game over |
| `lifeLossAsTerminated` | life loss ends episode as `terminated` (common older protocol) |
| `lifeLossAsTruncated` | life loss as `truncated` (caller may reset) |

Default for v1 adapter: **`gameOverOnly`** or **`lifeLossAsTerminated`** — **pick one at implementation and lock in tests**; document equivalence notes vs Gymnasium presets without promising identity.

### 5.4 Observation defaults (v1)

| Setting | Default (proposal) |
|---------|-------------------|
| Mode | **Grayscale** screen (full native resolution first **or** fixed resize — implementer chooses one and documents) |
| Dtype | `UInt8` in `MLXArray` if Box supports it; else `Float` scaled 0…1 — **must match ``BoxSpace`` capabilities** in core |
| Layout | Channel-last or channel-first — **pick one convention** and use it in all ALE envs |

**Preprocessing** (84×84, frame stack, float scale) should live in **wrappers** (existing `TransformObservation` or small ALE-specific wrappers in `RLXALE`), not forced inside the C++ boundary for every use case.

### 5.5 Seeding and determinism

- `reset(seed: non-nil)` → call ALE seed API when available; document no-op if unsupported for a pin.
- `spec.nondeterministic` set appropriately if sticky actions or incomplete seed coverage.
- No process-global MLX seed inside the adapter (core rule).

### 5.6 Errors

Map failures to ``EnvironmentError``:

| Situation | Error |
|-----------|--------|
| Missing/invalid ROM | `.configuration(...)` |
| Step before reset | `.notReset` (if not relying solely on OrderEnforcing) |
| Step after episode end | `.episodeEnded` |
| Use after close | `.closed` |
| Invalid action index | `.invalidAction(...)` |
| ALE internal failure | `.underlying(...)` or `.configuration(...)` |

---

## 6. Wrappers and factories

### 6.1 Recommended stacks

**Raw (debug):**

```text
ALEEnvironment(config)
```

**Training-oriented factory (adapter package):**

```text
RecordEpisodeStatistics?
→ TimeLimit?
→ [ALE preprocess: resize / gray / scale]
→ [FrameStack — new or TransformObservation-based]
→ OrderEnforcing?
→ ALEEnvironment
```

`PassiveEnvChecker` on image envs may be expensive; document as **dev-only**.

### 6.2 Registry ids (proposal)

Pattern: `ALE/<Game>-v0` (e.g. `ALE/Pong-v0`), registered only when `RLXALE` is linked and `registerALEDefaults` (or per-game register) is called.

Ids are **rlx-swift naming**, not a promise of Gymnasium string parity.

### 6.3 Frame stack

If needed for DQN-style agents:

- Prefer a small **`FrameStack` wrapper** in `RLXALE` or later promotion to `RLXWrappers` if reusable.
- Not a blocker for M0–M1 (single-frame obs OK for random agent / smoke).

---

## 7. Vectorization

| Phase | Approach |
|-------|----------|
| **Phase 1** | `SyncVectorEnv` / `AsyncVectorEnv` with `makeEnv: { ALEEnvironment(...) }` — N independent emulators |
| **Phase 2 (optional)** | Investigate Farama native Atari vector C++ API; expose only if profiling shows Phase 1 insufficient |

Partial failure: keep core vector semantics (one slot throws → whole batch fails) unless core changes later.

---

## 8. Build, ROMs, CI, license

### 8.1 Build

Document one primary path for **macOS arm64** (tier-1):

1. Build/install ALE via CMake (flags: SDL off for default).
2. Point Swift package at library + include (pkg-config, `ALE_ROOT`, or vendored `cmake` external project — choose at spike).
3. Optional: script `scripts/build-ale.sh` (implementation phase).

Linux tier-2: best-effort; same CMake story.

### 8.2 ROMs

- **Never commit copyrighted ROMs.**
- User sets `ALE_ROM_PATH` / config `romPath`.
- Tests skip with clear message if ROM missing.
- CI: optional job with a **legally redistributable** test ROM if one is identified; otherwise manual/nightly only.

### 8.3 License

- `rlx-swift` core remains **MIT**.
- Adapter must document **ALE license** and third-party notices.
- Design §27.3: copyleft adapters stay out of core targets — **satisfied** by optional `RLXALE` product.

### 8.4 CI matrix

| Job | ALE required? |
|-----|----------------|
| Existing macOS xcodebuild (core) | No |
| Linux smoke (core) | No |
| `RLXALE` unit/integration | Yes (optional workflow) |

---

## 9. Testing strategy

| Level | What |
|-------|------|
| Shim unit | C API: load ROM, reset, one act (requires ROM) |
| `Environment` | Spaces, reset obs shape/dtype, step reward finite, close |
| Contract | `checkEnvironment` where observation equality is feasible; else shape/dtype/smoke only |
| Vector | 2–4 envs, few steps, no crash |
| Golden | Optional soft snapshot (reward sum / obs hash) for one game+seed — platform-specific, not Gym oracle |

---

## 10. Documentation (when implementing)

| Artifact | Content |
|----------|---------|
| This file | Normative adapter design (update as decisions lock) |
| `Sources/RLXALE/*.docc` or module docs | API usage, factory, ROM setup |
| Root README | One short link: “Atari/ALE optional product” → this doc + build notes |
| `design.md` §24.3 | Pointer: “see `docs/ale-adapter-design.md`” |

Avoid duplicating full ALE manuals; link upstream ALE docs for emulator details.

---

## 11. Implementation plan (phased)

Do **not** start coding until this design is reviewed and open questions (§12) are decided or explicitly deferred.

### Phase 0 — Spike (throwaway OK)

**Exit criteria:** On a Mac with ALE built, a tiny C or C++ main (or Swift calling shim) loads one ROM, resets, steps, prints reward. No need for full `Environment` yet.

| Tasks | Notes |
|-------|--------|
| Pin ALE version | Record commit/tag here |
| CMake build SDL=OFF | Document commands |
| Prove screen buffer layout | H, W, channels, stride |
| Prove minimal action set | Count and mapping |
| Decide shim vs interop | Expect C shim |

**Estimate:** small (days), high risk reduction.

### Phase 1 — Package skeleton + shim

| Tasks |
|-------|
| Add optional `RLXALECXX` (or equivalent) target + `RLXALE` Swift target to `Package.swift` without breaking default builds when ALE absent (**or** clear `ALE_ROOT` requirement documented) |
| Implement C façade: create/destroy, load ROM, reset, act, screen copy, legal actions, game over, lives |
| Error codes → Swift `EnvironmentError` |

**Exit criteria:** Swift can call shim and complete one episode loop without full product polish.

### Phase 2 — `ALEEnvironment` + spaces

| Tasks |
|-------|
| `ALEConfig`, obs/action spaces |
| `reset` / `step` / `close` |
| Grayscale (or chosen default) `MLXArray` obs |
| Minimal action set |
| Lives policy locked |
| Unit tests gated on ROM |

**Exit criteria:** Conforms to `Environment`; random policy runs for N steps; `close` safe.

### Phase 3 — Product UX

| Tasks |
|-------|
| Registry helpers / factory with optional TimeLimit + OrderEnforcing |
| Example `ALERandomAgent` executable |
| DocC / README section for install + ROM |
| `checkEnvironment` or smoke path documented |

**Exit criteria:** External contributor with ALE+ROM can run example without reading C++.

### Phase 4 — Vector + preprocess (as needed)

| Tasks |
|-------|
| Document / test under `SyncVectorEnv` |
| Optional resize / float scale / frame stack wrappers |
| Profile copy cost screen → `MLXArray` |

**Exit criteria:** Stable multi-env collection for future DQN experiments.

### Phase 5 — Hardening (later)

| Tasks |
|-------|
| Sticky actions, full action set, RAM mode |
| More games via same config |
| Optional native ALE vector |
| Algorithms-repo DQN notebook/target (out of this package) |

---

## 12. Open questions (resolve before or during Phase 0–1)

| # | Question | Options | Proposal |
|---|----------|---------|----------|
| Q1 | Monorepo `RLXALE` vs separate repo | Monorepo first / separate only | **Monorepo optional product** |
| Q2 | Default observation | Gray full-res / gray 84×84 / RGB | **Gray + preprocess wrappers for 84×84** |
| Q3 | Default lives policy | gameOverOnly / lifeLossTerminated | **Decide in Phase 0 against one benchmark preset; document** |
| Q4 | How SPM finds ALE | `ALE_ROOT`, brew, vendored submodule | **Spike: `ALE_ROOT` + script; no submodule unless needed** |
| Q5 | First game | Pong / Breakout / Freeway | **Pong** (simple, common) |
| Q6 | `UInt8` vs `Float` obs in BoxSpace | Depends on core Box support | **Audit core `BoxSpace` in Phase 1; prefer uint8 if supported** |
| Q7 | Frameskip location | ALE internal vs Swift loop | **ALE frameskip config in `ALEConfig` for v1** |

---

## 13. Risks

| Risk | Mitigation |
|------|------------|
| SPM ↔ CMake friction | Phase 0 before package graph commitment; fallback binary target |
| Screen copy overhead | Measure; consider reusable buffer; vector N small at first |
| ROM/legal issues | No ROMs in git; skip tests |
| API drift in ALE | Pin version; shim isolates Swift from churn |
| Scope creep (full Gym Atari parity) | Phases 0–3 only one game + minimal actions |

---

## 14. Success criteria (adapter “v1 done”)

1. Optional product builds on macOS when ALE is installed and configured.
2. `ALEEnvironment` (or equivalent) implements `Environment` for at least **one** game.
3. Random agent example runs end-to-end with a user-supplied ROM.
4. Core packages remain buildable and tested **without** ALE.
5. This design doc updated with locked decisions (obs layout, lives policy, ALE pin).
6. No Atari/ALE dependency in `RLXCore` link line.

---

## 15. Relationship to future algorithms work

```text
rlx-swift (core + classic envs + vector)
    ↑
RLXALE (this adapter)
    ↑
rlx-swift-algorithms / user DQN (future) — not part of this design’s implementation phases
```

Proving **CartPole + algorithm** can still proceed in parallel; ALE is the **image/discrete** path, not a blocker for the first policy-gradient/PPO plumbing on classic control.

---

## 16. Document control

| Version | Date | Notes |
|---------|------|--------|
| 0.1 | 2026-07-01 | Initial draft: package boundary, mapping, phases 0–5, open questions |

**Next step after approval:** resolve §12 open questions (at least Q1, Q2, Q5, Q4), then execute **Phase 0 spike** on a follow-up branch (e.g. `feat/ale-spike`).

---

## Appendix A — Out-of-scope reminder from core design

From [`design.md`](../design.md):

- Training algorithms and buffers are **not** in rlx-swift core.
- Atari is listed under **out of scope for bootstrap**; this adapter **is** that future work, packaged optionally.
- Prefer in-process vector/env over multi-process for v1-style work.

## Appendix B — Illustrative call flow

```text
ALEEnvironment.step(action: Int)
  → validate action in DiscreteSpace / legal range
  → rlx_ale_act(handle, action_index)  // may apply frameskip inside
  → read reward, game_over, lives
  → rlx_ale_copy_screen(handle, buffer)
  → MLXArray from buffer (shape per config)
  → StepResult(obs, reward, terminated, truncated, info)
```
