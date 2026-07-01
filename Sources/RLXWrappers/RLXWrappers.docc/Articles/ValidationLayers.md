# Validation layers

## Overview

`rlx-swift` separates **when** and **how** environments are checked so production paths stay
fast while development stays strict.

### ``OrderEnforcing``

Tracks whether `reset` has been called and whether the episode has ended. Throws:

- ``EnvironmentError/notReset``
- ``EnvironmentError/episodeEnded``

It does **not** call `contains` on observations or actions.

### ``PassiveEnvChecker``

On every `reset` / `step`:

1. Observation ∈ observation space
2. Action ∈ action space (checked **before** `inner.step`)
3. Reward is finite (`Float.isFinite`)

It does **not** change dynamics (no clipping, no autoreset). Failures use
``EnvironmentError/invalidObservation(_:)``, ``EnvironmentError/invalidAction(_:)``, or
``EnvironmentError/configuration(_:)`` for non-finite rewards.

### ``checkEnvironment`` (`RLXTesting`)

A **function**, not a wrapper: runs many episodes, optional determinism comparison, close
idempotency. Use in unit tests and CI. Prefer a factory `() -> Env` so two instances can be built.

### Recommended developer stack

```swift
PassiveEnvChecker(OrderEnforcing(TimeLimit(MyEnv(), maxEpisodeSteps: n)))
```

Run ``checkEnvironment`` in tests on the **unwrapped or lightly wrapped** env according to
what you want to assert (order enforcement can be enabled via
``CheckEnvironmentOptions/enforceOrder``).
