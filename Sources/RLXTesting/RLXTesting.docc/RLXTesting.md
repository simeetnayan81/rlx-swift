# ``RLXTesting``

Contract tests and harnesses for ``Environment`` implementations.

@Metadata {
    @TechnologyRoot
}

## Overview

### ``checkEnvironment``

Exercises (see `design.md` §20.3):

1. Space sample + `contains`
2. Reset produces in-space observation
3. Random policy for N episodes (finite rewards, in-space obs)
4. Optional post-terminal order check via ``OrderEnforcing``
5. Determinism pair when `spec.nondeterministic` is not true
6. Close idempotency

Use a **factory** `() -> Env` so the harness can construct independent instances.

Pair with ``PassiveEnvChecker`` in **manual** debugging; keep unit tests explicit about which
wrappers they include.

## Topics

### Harness

- ``checkEnvironment(_:options:)``
- ``CheckEnvironmentOptions``
- ``CheckEnvironmentError``

## See also

- `RLXWrappers` validation layers article
- `design.md` §20, §23
