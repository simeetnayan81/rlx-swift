// DummyEnv — fixed-length deterministic debugging env (design.md §24.2, PR-07).

import RLXCore

/// Toy discrete environment with fixed-length episodes and deterministic rewards.
///
/// - Observation / action: `Int` via ``DiscreteSpace``.
/// - Dynamics: `obs' = (obs + action) % observationN`, `reward = Float(action)`.
/// - Episode ends with `terminated = true` after `episodeLength` steps (not truncation;
///   TimeLimit / `truncated` is PR-08).
/// - Enforces lifecycle: `notReset`, `episodeEnded`, `closed`.
/// - `close()` is idempotent; further `reset` / `step` throw ``EnvironmentError/closed``.
public final class DummyEnv: Environment {
    public typealias Observation = Int
    public typealias Action = Int
    public typealias ObservationSpace = DiscreteSpace
    public typealias ActionSpace = DiscreteSpace

    public let observationSpace: DiscreteSpace
    public let actionSpace: DiscreteSpace
    /// Steps per episode before `terminated` (must be > 0).
    public let episodeLength: Int
    public let spec: EnvSpec?

    private var obs: Int = 0
    private var stepsInEpisode: Int = 0
    private var hasReset = false
    private var episodeOver = false
    private var isClosed = false

    /// - Parameters:
    ///   - observationN: Size of observation discrete space (default 5).
    ///   - actionN: Size of action discrete space (default 5).
    ///   - episodeLength: Steps until termination (default 10).
    public init(
        observationN: Int = 5,
        actionN: Int = 5,
        episodeLength: Int = 10
    ) {
        precondition(observationN > 0, "observationN must be > 0")
        precondition(actionN > 0, "actionN must be > 0")
        precondition(episodeLength > 0, "episodeLength must be > 0")
        self.observationSpace = DiscreteSpace(n: observationN)
        self.actionSpace = DiscreteSpace(n: actionN)
        self.episodeLength = episodeLength
        self.spec = EnvSpec(
            id: "DummyEnv-v0",
            maxEpisodeSteps: episodeLength,
            nondeterministic: false,
            version: 1
        )
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Int> {
        if isClosed { throw EnvironmentError.closed }
        // Seed accepted for API / determinism checks; dynamics are fully deterministic
        // from the reset observation and action sequence (no RNG draws).
        _ = seed
        _ = options
        hasReset = true
        episodeOver = false
        stepsInEpisode = 0
        obs = observationSpace.start
        return ResetResult(observation: obs)
    }

    public func step(_ action: Int) throws -> StepResult<Int> {
        if isClosed { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        if episodeOver { throw EnvironmentError.episodeEnded }
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction("action \(action) out of space")
        }
        let n = observationSpace.n
        let start = observationSpace.start
        // Map into [start, start+n) with modular arithmetic relative to start.
        let offset = ((obs - start) + action) % n
        let positiveOffset = offset >= 0 ? offset : offset + n
        obs = start + positiveOffset
        stepsInEpisode += 1
        let terminated = stepsInEpisode >= episodeLength
        if terminated { episodeOver = true }
        return StepResult(
            observation: obs,
            reward: Float(action),
            terminated: terminated,
            truncated: false
        )
    }

    public func close() throws {
        isClosed = true
    }
}
