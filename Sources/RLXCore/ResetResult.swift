// ResetResult — outcome of Environment.reset (design.md §11.1).

/// Value returned by `Environment.reset`.
///
/// Does **not** include reward, terminated, or truncated — those belong only
/// on `StepResult`.
public struct ResetResult<Observation> {
    /// Initial observation; must lie in the env's observation space.
    public var observation: Observation
    /// Diagnostics; may be empty.
    public var info: Info

    public init(observation: Observation, info: Info = Info()) {
        self.observation = observation
        self.info = info
    }
}

extension ResetResult: Equatable where Observation: Equatable {}
extension ResetResult: Sendable where Observation: Sendable {}
