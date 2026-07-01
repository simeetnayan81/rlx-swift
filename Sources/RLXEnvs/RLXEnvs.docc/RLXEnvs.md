# ``RLXEnvs``

Reference and debugging environments for validating the `rlx-swift` API.

@Metadata {
    @TechnologyRoot
}

## Overview

Ship a **small** bootstrap set — not a competitor to specialized simulators:

| ID / type | Role |
|-----------|------|
| ``DummyEnv`` | Fixed-length discrete toy env for tests and smoke |
| ``CartPole`` / CartPole-v1 | Classic control, tensor-native |
| ``Pendulum`` / Pendulum-v1 | Continuous actions (`Box`) |

Factories often install a default wrapper stack (order + time limit). Prefer registry
``EnvironmentRegistry/make`` when integrating by string id.

## Topics

### Debugging

- ``DummyEnv``

### Classic control

- ``CartPole``
- ``Pendulum``

### Registration

Registration helpers run from this module so ids are available after import.

## See also

- `design.md` §24 (reference environments)
- `RLXTesting` for ``checkEnvironment``
