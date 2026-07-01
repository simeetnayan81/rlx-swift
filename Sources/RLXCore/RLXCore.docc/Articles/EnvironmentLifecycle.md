# Environment lifecycle

## Overview

An ``Environment`` is **class-bound** (`AnyObject`): it owns mutable episode state.
Construction does **not** start an episode.

```text
construct → reset → step* → (terminated|truncated) → reset → …
                              ↘ close (anytime; further reset/step throw .closed)
```

### Rules (normative)

1. **First legal call is `reset`.** Calling `step` before `reset` should fail
   (``EnvironmentError/notReset``). Core documents the rule; ``OrderEnforcing``
   (`RLXWrappers`) enforces it for envs that do not.
2. **After `terminated || truncated`, the next legal call is `reset`** before another `step`
   (``EnvironmentError/episodeEnded``), unless a **vector** autoreset policy applies
   (`RLXVector` — documented separately).
3. **`close` is idempotent** at the concrete env’s discretion; further interaction should
   throw ``EnvironmentError/closed``.
4. **Reward is always `Float`** on ``StepResult`` (MLX float32 alignment). There is no
   associated reward type on ``Environment``.
5. **Seeding** happens via `reset(seed:)` — not a process-global MLX seed inside library code.

### Minimal interaction loop

```swift
let env: some Environment = /* … */
var rng = SplitMix64(seed: 0)
_ = try env.reset(seed: 42)
var done = false
while !done {
    let action = env.actionSpace.sample(using: &rng)
    let step = try env.step(action)
    done = step.done  // terminated || truncated
}
try env.close()
```

### Termination vs truncation

| Flag | Meaning |
|------|---------|
| `terminated` | Task / MDP end (success, failure, absorbing state) |
| `truncated` | External stop (time limit, etc.) — often still bootstrapable |

Wrappers such as ``TimeLimit`` (`RLXWrappers`) set `truncated` without inventing a task
terminal. Do not collapse both into a single `done` in new APIs; use ``StepResult/done``
only as a convenience for “episode over.”

### Related design sections

`design.md` §8.2 (Environment), §11 (lifecycle), §12 (termination vs truncation), §21 (sync API).
