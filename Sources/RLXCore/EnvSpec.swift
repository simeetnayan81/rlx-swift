// EnvSpec — immutable metadata for an environment kind (design.md §8.6, §14.2).

/// Catalog card for an environment **kind** (not one live instance).
///
/// Instance knobs (dt, gravity, …) belong on env-specific config types; the registry
/// (PR-10) pairs ``EnvSpec`` with factories that produce ``AnyEnvironment``.
public struct EnvSpec: Sendable, Equatable, Hashable, Codable {
    /// Stable id, e.g. `"CartPole-v1"`.
    public var id: String
    /// Hint for episode length / ``TimeLimit`` (optional).
    public var maxEpisodeSteps: Int?
    /// Optional “solved” return threshold.
    public var rewardThreshold: Double?
    /// If `true`, same seed may not reproduce bit-identical trajectories.
    public var nondeterministic: Bool
    /// Default render mode for `make` when unspecified.
    public var defaultRenderMode: RenderMode?
    /// Bump when dynamics or obs/action contract change intentionally.
    public var version: Int

    public init(
        id: String,
        maxEpisodeSteps: Int? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false,
        defaultRenderMode: RenderMode? = nil,
        version: Int = 1
    ) {
        self.id = id
        self.maxEpisodeSteps = maxEpisodeSteps
        self.rewardThreshold = rewardThreshold
        self.nondeterministic = nondeterministic
        self.defaultRenderMode = defaultRenderMode
        self.version = version
    }
}
