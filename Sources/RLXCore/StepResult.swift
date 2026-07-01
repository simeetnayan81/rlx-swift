// StepResult — outcome of Environment.step (design.md §11.2, §7.3.1).

/// Outcome of a successful ``Environment/step(_:)``.
///
/// Describes one transition `(s, a) → (s', r, terminated, truncated)` plus diagnostics.
///
/// - **Reward** is locked to `Float` (IEEE-754 binary32) so training code aligns with
///   MLX float32 defaults without an associated reward type on every env/wrapper.
/// - Prefer inspecting ``terminated`` and ``truncated`` separately (bootstrap vs absorb).
///   Use ``done`` only when you need “episode over for any reason.”
/// - After ``done`` is `true`, the next legal call is ``Environment/reset(seed:options:)``
///   unless a vector autoreset policy applies (`RLXVector`).
public struct StepResult<Observation> {
    /// Successor observation `s'` (still the terminal state when the episode ends).
    public var observation: Observation
    /// Scalar reward for the transition just taken; must be finite in well-formed envs.
    public var reward: Float
    /// Task / MDP end (success, failure, absorbing state).
    public var terminated: Bool
    /// External cutoff (time limit, resource bound, …) — often still bootstrapable.
    public var truncated: Bool
    /// Side-channel diagnostics (time-limit markers, episode stats, vector finals, …).
    public var info: Info

    /// - Parameters:
    ///   - observation: Successor observation.
    ///   - reward: Transition reward (`Float`, prefer finite).
    ///   - terminated: MDP terminal flag.
    ///   - truncated: External truncation flag.
    ///   - info: Optional diagnostics (default empty).
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

    /// `true` when the episode ended for **any** reason (`terminated || truncated`).
    ///
    /// Prefer the separate flags when learning targets differ for terminate vs truncate.
    public var done: Bool {
        terminated || truncated
    }
}

extension StepResult: Equatable where Observation: Equatable {}
extension StepResult: Sendable where Observation: Sendable {}
