// InfoKeys — stable diagnostic key strings (design.md §14.1, PR-08).

// No RLXCore import required — string constants only.

/// Compile-time constants for documented `Info` keys.
///
/// Prefer these over string literals so stacked wrappers do not drift on typos.
public enum InfoKeys: Sendable {
    /// Set by ``TimeLimit`` when this step hit the step limit (`truncated` is the primary signal).
    public static let timeLimitTruncated = "TimeLimit.truncated"

    /// Nested bag set by ``RecordEpisodeStatistics`` on episode end.
    public static let episode = "episode"

    /// Cumulative return for the completed episode (under ``episode``).
    public static let episodeReturn = "r"

    /// Step count for the completed episode (under ``episode``).
    public static let episodeLength = "l"

    /// Optional wall-clock seconds for the completed episode (under ``episode``).
    public static let episodeTime = "t"

    /// Terminal observation when vector autoreset already returned the next obs (PR-13+).
    public static let finalObservation = "final_observation"

    /// Terminal info in the same situation as ``finalObservation`` (PR-13+).
    public static let finalInfo = "final_info"
}
