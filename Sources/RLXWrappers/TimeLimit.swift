// TimeLimit — truncate episodes after a fixed step count (design.md §12, §15.2, PR-08).

import RLXCore

/// Counts steps since the last ``reset`` and sets ``StepResult/truncated`` when the limit is hit.
///
/// Does **not** set ``StepResult/terminated`` for the limit alone (task MDP stays clean).
/// On limit truncation, also sets ``InfoKeys/timeLimitTruncated`` in `info`.
///
/// If the inner env terminates on the same step as the limit, both flags may be true.
public final class TimeLimit<Inner: Environment>: Environment, EnvironmentWrapper {
    public typealias Observation = Inner.Observation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner
    /// Maximum steps per episode after each reset (must be > 0).
    public let maxEpisodeSteps: Int

    private var elapsedSteps = 0

    /// - Parameters:
    ///   - inner: Wrapped environment.
    ///   - maxEpisodeSteps: Steps before forcing `truncated` (must be positive).
    public init(_ inner: Inner, maxEpisodeSteps: Int) {
        precondition(maxEpisodeSteps > 0, "maxEpisodeSteps must be > 0")
        self.inner = inner
        self.maxEpisodeSteps = maxEpisodeSteps
    }

    public var observationSpace: ObservationSpace { inner.observationSpace }
    public var actionSpace: ActionSpace { inner.actionSpace }
    public var spec: EnvSpec? { inner.spec }

    public var unwrapped: AnyEnvironment {
        AnyEnvironment(inner)
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Observation> {
        let result = try inner.reset(seed: seed, options: options)
        elapsedSteps = 0
        return result
    }

    public func step(_ action: Action) throws -> StepResult<Observation> {
        var result = try inner.step(action)
        elapsedSteps += 1
        if elapsedSteps >= maxEpisodeSteps {
            result.truncated = true
            result.info[InfoKeys.timeLimitTruncated] = .bool(true)
        }
        return result
    }

    public func close() throws {
        try inner.close()
    }
}
