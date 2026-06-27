// StepResult — outcome of Environment.step (design.md §11.2, §7.3.1).

/// Value returned by `Environment.step`.
///
/// Reward is locked to `Float` (IEEE-754 binary32) for single-env steps so
/// training code aligns with MLX float32 defaults without an associated-type
/// explosion on every env/wrapper.
public struct StepResult<Observation> {
    /// Successor observation `s'`.
    public var observation: Observation
    /// Scalar reward for the transition `(s, a, s')` just taken.
    public var reward: Float
    /// `true` when an MDP terminal / absorbing state was reached.
    public var terminated: Bool
    /// `true` when an external cutoff ended the episode (time limit, bounds, …).
    public var truncated: Bool
    /// Extra diagnostics; may be empty.
    public var info: Info

    public init(
        observation: Observation,
        reward: Float,
        terminated: Bool,
        truncated: Bool,
        info: Info = Info()
    ) {
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
    }

    /// Whether the episode ended for any reason (`terminated || truncated`).
    public var done: Bool {
        terminated || truncated
    }
}

extension StepResult: Equatable where Observation: Equatable {}
extension StepResult: Sendable where Observation: Sendable {}
