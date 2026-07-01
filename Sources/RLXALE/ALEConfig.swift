/// Configuration for ``ALEEnvironment`` (docs/ale-adapter-design.md).

public struct ALEConfig: Sendable, Equatable {
    /// Filesystem path to an Atari ROM (user-supplied; not shipped with rlx-swift).
    public var romPath: String
    /// Screen observation layout.
    public var observationType: ALEObservationType
    /// ALE `frame_skip` (applied before ROM load).
    public var frameSkip: Int
    /// Sticky-action probability (0 = off). ALE `repeat_action_probability`.
    public var repeatActionProbability: Float
    /// How life loss maps to episode flags.
    public var livesPolicy: ALELivesPolicy
    /// Optional seed applied via ALE `random_seed` before ROM load; also used on first reset if set.
    public var seed: Int32?

    public init(
        romPath: String,
        observationType: ALEObservationType = .grayscale,
        frameSkip: Int = 4,
        repeatActionProbability: Float = 0,
        livesPolicy: ALELivesPolicy = .gameOverOnly,
        seed: Int32? = nil
    ) {
        precondition(!romPath.isEmpty, "romPath must be non-empty")
        precondition(frameSkip >= 1, "frameSkip must be >= 1")
        precondition(repeatActionProbability >= 0 && repeatActionProbability <= 1)
        self.romPath = romPath
        self.observationType = observationType
        self.frameSkip = frameSkip
        self.repeatActionProbability = repeatActionProbability
        self.livesPolicy = livesPolicy
        self.seed = seed
    }
}

public enum ALEObservationType: String, Sendable, Equatable, CaseIterable {
    /// Single-channel screen, shape `[H, W]`, float32 in `0...255`.
    case grayscale
    /// Interleaved RGB, shape `[H, W, 3]`, float32 in `0...255`.
    case rgb
}

/// Maps ALE life loss / game over to ``StepResult`` flags.
public enum ALELivesPolicy: String, Sendable, Equatable, CaseIterable {
    /// `terminated` only when ALE reports game over.
    case gameOverOnly
    /// Also treat a decrease in lives as `terminated`.
    case lifeLossAsTerminated
}
