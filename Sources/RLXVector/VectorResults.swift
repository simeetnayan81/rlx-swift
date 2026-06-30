// Batched reset/step results for vector environments (design.md §8.4, PR-13).

import RLXCore

/// Outcome of ``SyncVectorEnv/reset(seed:options:)``.
/// Not `Sendable`: observations are type-erased `Any` (may hold `MLXArray`).
public struct VectorResetResult {
    /// One observation per sub-environment (type-erased).
    public var observations: [Any]
    /// Per-env info bags (same count as observations).
    public var infos: [Info]

    public init(observations: [Any], infos: [Info]) {
        precondition(observations.count == infos.count)
        self.observations = observations
        self.infos = infos
    }
}

/// Outcome of ``SyncVectorEnv/step(_:)``.
/// Not `Sendable`: observations are type-erased `Any` (may hold `MLXArray`).
public struct VectorStepResult {
    public var observations: [Any]
    public var rewards: [Float]
    public var terminateds: [Bool]
    public var truncateds: [Bool]
    public var infos: [Info]

    public init(
        observations: [Any],
        rewards: [Float],
        terminateds: [Bool],
        truncateds: [Bool],
        infos: [Info]
    ) {
        let n = observations.count
        precondition(rewards.count == n && terminateds.count == n && truncateds.count == n && infos.count == n)
        self.observations = observations
        self.rewards = rewards
        self.terminateds = terminateds
        self.truncateds = truncateds
        self.infos = infos
    }

    /// Per-index `terminated || truncated`.
    public func dones() -> [Bool] {
        zip(terminateds, truncateds).map { $0 || $1 }
    }
}
