// PassiveEnvChecker — opt-in value validation on reset/step (design.md §15.2, §20.2, PR-15).

import RLXCore

/// Opt-in wrapper that validates observations, actions, and rewards without changing dynamics.
///
/// ## What it checks
///
/// | Call | Checks |
/// |------|--------|
/// | ``reset(seed:options:)`` | Returned observation is in ``observationSpace`` |
/// | ``step(_:)`` | Action is in ``actionSpace`` **before** calling `inner`; returned observation is in ``observationSpace``; reward is finite |
///
/// Failures throw ``EnvironmentError/invalidObservation(_:)``, ``EnvironmentError/invalidAction(_:)``,
/// or ``EnvironmentError/configuration(_:)`` (non-finite reward).
///
/// ## Passive vs other layers
///
/// - **Passive:** does not clip, reshape, autoreset, or alter `terminated` / `truncated`.
/// - Contrast with ``OrderEnforcing`` (lifecycle / call order only).
/// - Contrast with ``checkEnvironment`` in `RLXTesting` (multi-episode harness for tests, not a live wrapper).
///
/// Cost is **medium** (space `contains` on every transition). Prefer in development and debugging;
/// omit on the hottest production collection paths if profiling shows it matters.
///
/// ## Example
///
/// ```swift
/// let env = PassiveEnvChecker(
///     OrderEnforcing(
///         TimeLimit(MyEnv(), maxEpisodeSteps: 200)
///     )
/// )
/// ```
///
/// > Design reference: `design.md` §15.2 (wrapper set), §20.2 (validation layers).
public final class PassiveEnvChecker<Inner: Environment>: Environment, EnvironmentWrapper {
    public typealias Observation = Inner.Observation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner

    public init(_ inner: Inner) {
        self.inner = inner
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
        try validateObservation(result.observation, context: "reset")
        return result
    }

    public func step(_ action: Action) throws -> StepResult<Observation> {
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction(
                "PassiveEnvChecker: action not in actionSpace (\(String(describing: action)))"
            )
        }
        let result = try inner.step(action)
        try validateObservation(result.observation, context: "step")
        guard result.reward.isFinite else {
            throw EnvironmentError.configuration(
                "PassiveEnvChecker: reward must be finite, got \(result.reward)"
            )
        }
        return result
    }

    public func close() throws {
        try inner.close()
    }

    private func validateObservation(_ observation: Observation, context: String) throws {
        guard observationSpace.contains(observation) else {
            throw EnvironmentError.invalidObservation(
                "PassiveEnvChecker: \(context) observation not in observationSpace (\(String(describing: observation)))"
            )
        }
    }
}
