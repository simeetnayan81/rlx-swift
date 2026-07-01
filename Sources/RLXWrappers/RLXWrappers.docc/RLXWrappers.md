# ``RLXWrappers``

Composable **environment adapters**: lifecycle enforcement, time limits, transforms, statistics,
and opt-in **value validation**.

@Metadata {
    @TechnologyRoot
}

## Overview

Wrappers implement ``Environment`` and typically ``EnvironmentWrapper``, holding an `inner`
environment and overriding only what they change. Stack them outside-in:

```swift
let env = PassiveEnvChecker(
    RecordEpisodeStatistics(
        TimeLimit(
            OrderEnforcing(MyEnv()),
            maxEpisodeSteps: 500
        )
    )
)
```

### Validation cost tiers (`design.md` §20.2)

| Layer | Role | Cost |
|-------|------|------|
| Debug asserts | Internal invariants | Cheap |
| ``OrderEnforcing`` | Call order (`reset` / `step`) | Cheap |
| ``PassiveEnvChecker`` | `contains` + finite reward each transition | Medium |
| ``checkEnvironment`` (`RLXTesting`) | Multi-episode contract suite | Test-time |

Use ``PassiveEnvChecker`` while developing custom envs; measure before leaving it on
ultra-hot collection loops.

## Topics

### Getting started

- <doc:ValidationLayers>
- <doc:WrapperComposition>
- <doc:CustomEnvironmentGuide>

### Lifecycle and limits

- ``OrderEnforcing``
- ``TimeLimit``
- ``RecordEpisodeStatistics``

### Validation

- ``PassiveEnvChecker``

### Action / observation / reward transforms

- ``ClipAction``
- ``RescaleAction``
- ``TransformObservation``
- ``TransformReward``

### Keys and protocol

- ``InfoKeys``
- ``EnvironmentWrapper``

## See also

- Repository `Documentation/DeveloperGuide.md`
- `design.md` §15 (wrappers), §20 (validation)
- `RLXCore` DocC *Architecture for developers* and lifecycle articles
- Example: `swift run RandomAgentDemo`
