import RLXCore
import Foundation

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
    /// Optional seed applied via ALE `random_seed` before ROM load.
    public var seed: Int32?
    /// Request ALE SDL window (only if ALE was built with SDL support).
    public var displayScreen: Bool
    /// Request ALE sound (only if ALE was built with sound support).
    public var sound: Bool

    public init(
        romPath: String,
        observationType: ALEObservationType = .grayscale,
        frameSkip: Int = 4,
        repeatActionProbability: Float = 0,
        livesPolicy: ALELivesPolicy = .gameOverOnly,
        seed: Int32? = nil,
        displayScreen: Bool = false,
        sound: Bool = false
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
        self.displayScreen = displayScreen
        self.sound = sound
    }

    /// Resolve `directory/<game>.bin` (case-sensitive filename).
    public static func resolveROMPath(game: String, directory: String) throws -> String {
        let base = (game as NSString).deletingPathExtension
        let name = base.isEmpty ? game : base
        let candidates = [
            (directory as NSString).appendingPathComponent("\(name).bin"),
            (directory as NSString).appendingPathComponent(name),
            (directory as NSString).appendingPathComponent("\(name.lowercased()).bin"),
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            return path
        }
        throw EnvironmentError.configuration(
            "No ROM for game '\(game)' in \(directory). Tried: \(candidates.joined(separator: ", "))"
        )
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
