// ResetResult — outcome of Environment.reset (design.md §11.1).

/// Outcome of a successful ``Environment/reset(seed:options:)``.
///
/// Carries only the **initial** observation and optional diagnostics. Reward,
/// `terminated`, and `truncated` appear exclusively on ``StepResult`` after a `step`.
///
/// Custom envs should ensure `observation` lies in ``Environment/observationSpace``
/// (`PassiveEnvChecker` in `RLXWrappers` and `checkEnvironment` in `RLXTesting` enforce this when used).
public struct ResetResult<Observation> {
    /// Initial observation of the episode (`s₀`); must satisfy observation-space membership.
    public var observation: Observation
    /// Side-channel diagnostics (often empty at reset). Prefer scoped keys (see `InfoKeys` in `RLXWrappers`).
    public var info: Info

    /// - Parameters:
    ///   - observation: Episode start observation.
    ///   - info: Optional diagnostics bag (default empty).
    public init(observation: Observation, info: Info = Info()) {
        self.observation = observation
        self.info = info
    }
}

extension ResetResult: Equatable where Observation: Equatable {}
extension ResetResult: Sendable where Observation: Sendable {}
