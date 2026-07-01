# Implement a custom environment

Build a minimal ``Environment`` in about fifty lines, validate it, and run a random agent.

## 1. Define spaces and state

```swift
import RLXCore

public final class WalkEnv: Environment {
    public typealias Observation = Int
    public typealias Action = Int
    public typealias ObservationSpace = DiscreteSpace
    public typealias ActionSpace = DiscreteSpace

    public let observationSpace = DiscreteSpace(n: 5)  // positions 0…4
    public let actionSpace = DiscreteSpace(n: 3)       // -1, 0, +1 encoded as 0,1,2
    public let goal: Int
    public var spec: EnvSpec? {
        EnvSpec(id: "WalkEnv-v0", maxEpisodeSteps: 20, nondeterministic: false, version: 1)
    }

    private var position = 0
    private var steps = 0
    private var hasReset = false
    private var ended = false
    private var closed = false

    public init(goal: Int = 4) {
        self.goal = goal
    }
```

## 2. Implement `reset` and `step`

```swift
    public func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        if closed { throw EnvironmentError.closed }
        _ = seed; _ = options
        hasReset = true
        ended = false
        steps = 0
        position = 0
        return ResetResult(observation: position)
    }

    public func step(_ action: Int) throws -> StepResult<Int> {
        if closed { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        if ended { throw EnvironmentError.episodeEnded }
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction("\(action)")
        }
        let delta = action - 1          // 0,1,2 → -1,0,+1
        position = min(4, max(0, position + delta))
        steps += 1
        let terminated = position == goal
        let reward: Float = terminated ? 1 : -0.01
        ended = terminated
        return StepResult(
            observation: position,
            reward: reward,
            terminated: terminated,
            truncated: false
        )
    }

    public func close() throws { closed = true }
}
```

## 3. Validate in tests

```swift
import RLXTesting
import RLXWrappers

try checkEnvironment({ WalkEnv() })
let guarded = PassiveEnvChecker(OrderEnforcing(WalkEnv()))
_ = try guarded.reset(seed: 0)
```

## 4. Run a random policy

Sample from `actionSpace` with ``SplitMix64`` (see `Examples/RandomAgentDemo` in the repo, or
`swift run RandomAgentDemo`).

## Checklist

- [ ] `Observation` / `Action` match space `Value` types
- [ ] Reward is `Float` and finite
- [ ] Lifecycle errors (`notReset`, `episodeEnded`, `closed`) as appropriate
- [ ] Prefer explicit `reset(seed:)` over global MLX seeding
- [ ] Pass ``checkEnvironment``; use ``PassiveEnvChecker`` while debugging
- [ ] Read `design.md` §8, §11, §12, §15, §20 before changing contracts

## Related design sections

`design.md` Appendix A (protocol sketch), §24 (reference envs), §28 PR plan.
