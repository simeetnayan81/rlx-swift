// ALEEnvironment — Atari via ALE C++ shim (docs/ale-adapter-design.md).

import Foundation
import RLXALECXX
import MLX
import RLXCore

/// Atari 2600 environment backed by the Arcade Learning Environment (optional C++ dependency).
///
/// Build with `ALE_ROOT` pointing at an ALE install to link the real library; without it,
/// construction throws ``EnvironmentError/configuration(_:)`` explaining how to enable ALE.
///
/// - Observation: ``MLXArray`` float32 grayscale `[H,W]` or RGB `[H,W,3]` with values in `0...255`.
/// - Action: index into ALE **minimal** action set (``DiscreteSpace``).
/// - ROMs are **not** shipped; pass a path in ``ALEConfig/romPath``.
public final class ALEEnvironment: Environment {
    public typealias Observation = MLXArray
    public typealias Action = Int
    public typealias ObservationSpace = BoxSpace
    public typealias ActionSpace = DiscreteSpace

    public let config: ALEConfig
    public private(set) var observationSpace: BoxSpace
    public private(set) var actionSpace: DiscreteSpace
    public let spec: EnvSpec?

    private var handle: OpaquePointer?
    private var height = 0
    private var width = 0
    private var hasReset = false
    private var episodeOver = false
    private var isClosed = false
    private var lastLives = 0
    private var grayBuffer: [UInt8] = []
    private var rgbBuffer: [UInt8] = []

    /// - Throws: ``EnvironmentError/configuration(_:)`` if ALE is not linked or ROM load fails.
    public init(config: ALEConfig) throws {
        guard ALEBridge.isLinked() else {
            throw EnvironmentError.configuration(
                "ALE C++ library not linked. Set ALE_ROOT to your ALE install prefix and rebuild (see docs/ale-adapter-design.md)."
            )
        }
        self.config = config
        self.observationSpace = BoxSpace(low: 0, high: 255, shape: [1, 1], dtype: .float32)
        self.actionSpace = DiscreteSpace(n: 1)
        self.spec = EnvSpec(
            id: "ALE/Custom-v0",
            maxEpisodeSteps: nil,
            nondeterministic: config.repeatActionProbability > 0,
            version: 0
        )

        guard let raw = rlx_ale_create() else {
            throw EnvironmentError.configuration("rlx_ale_create failed")
        }
        self.handle = raw

        if let seed = config.seed {
            try ALEBridge.check(rlx_ale_set_int(raw, "random_seed", seed), context: "set random_seed")
        }
        try ALEBridge.check(rlx_ale_set_int(raw, "frame_skip", Int32(config.frameSkip)), context: "set frame_skip")
        try ALEBridge.check(
            rlx_ale_set_float(raw, "repeat_action_probability", config.repeatActionProbability),
            context: "set repeat_action_probability"
        )
        // Disable display / sound for headless collection.
        _ = rlx_ale_set_bool(raw, "display_screen", 0)
        _ = rlx_ale_set_bool(raw, "sound", 0)

        try config.romPath.withCString { cPath in
            try ALEBridge.check(rlx_ale_load_rom(raw, cPath), context: "loadROM")
        }

        self.height = Int(rlx_ale_screen_height(raw))
        self.width = Int(rlx_ale_screen_width(raw))
        let nActions = Int(rlx_ale_minimal_action_count(raw))
        guard height > 0, width > 0, nActions > 0 else {
            rlx_ale_destroy(raw)
            self.handle = nil
            throw EnvironmentError.configuration("ALE ROM loaded but invalid screen/action metadata")
        }

        switch config.observationType {
        case .grayscale:
            self.observationSpace = BoxSpace(low: 0, high: 255, shape: [height, width], dtype: .float32)
            self.grayBuffer = [UInt8](repeating: 0, count: height * width)
        case .rgb:
            self.observationSpace = BoxSpace(low: 0, high: 255, shape: [height, width, 3], dtype: .float32)
            self.rgbBuffer = [UInt8](repeating: 0, count: height * width * 3)
        }
        self.actionSpace = DiscreteSpace(n: nActions)
        self.lastLives = Int(rlx_ale_lives(raw))
    }

    deinit {
        if let handle {
            rlx_ale_destroy(handle)
        }
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<MLXArray> {
        _ = options
        try ensureOpen()
        guard let handle else { throw EnvironmentError.closed }

        // ALE applies random_seed most reliably before loadROM; for mid-session reseed we set and note limits.
        if let seed {
            let clamped = Int32(clamping: seed % UInt64(Int32.max))
            try ALEBridge.check(rlx_ale_set_int(handle, "random_seed", clamped), context: "reset seed")
        }
        try ALEBridge.check(rlx_ale_reset(handle), context: "reset_game")
        hasReset = true
        episodeOver = false
        lastLives = Int(rlx_ale_lives(handle))
        let obs = try captureObservation(handle)
        return ResetResult(observation: obs)
    }

    public func step(_ action: Int) throws -> StepResult<MLXArray> {
        try ensureOpen()
        guard let handle else { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        if episodeOver { throw EnvironmentError.episodeEnded }
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction("action \(action) not in 0..<\(actionSpace.n)")
        }

        var reward: Float = 0
        try ALEBridge.check(
            rlx_ale_act_minimal_index(handle, Int32(action), &reward),
            context: "act"
        )

        let livesNow = Int(rlx_ale_lives(handle))
        let gameOver = rlx_ale_game_over(handle) != 0
        var terminated = gameOver
        let truncated = false

        if config.livesPolicy == .lifeLossAsTerminated, !gameOver, livesNow < lastLives {
            terminated = true
        }
        lastLives = livesNow

        if terminated || truncated {
            episodeOver = true
        }

        let obs = try captureObservation(handle)
        var info = Info()
        info["ale.lives"] = .int(livesNow)
        return StepResult(
            observation: obs,
            reward: reward,
            terminated: terminated,
            truncated: truncated,
            info: info
        )
    }

    public func close() throws {
        if isClosed { return }
        isClosed = true
        if let handle {
            rlx_ale_destroy(handle)
            self.handle = nil
        }
    }

    private func ensureOpen() throws {
        if isClosed { throw EnvironmentError.closed }
    }

    private func captureObservation(_ handle: OpaquePointer) throws -> MLXArray {
        switch config.observationType {
        case .grayscale:
            try grayBuffer.withUnsafeMutableBytes { raw in
                let ptr = raw.bindMemory(to: UInt8.self).baseAddress!
                try ALEBridge.check(
                    rlx_ale_copy_screen_gray(handle, ptr, Int32(grayBuffer.count)),
                    context: "getScreenGrayscale"
                )
            }
            let floats = grayBuffer.map { Float($0) }
            return MLXArray(floats, [height, width])
        case .rgb:
            try rgbBuffer.withUnsafeMutableBytes { raw in
                let ptr = raw.bindMemory(to: UInt8.self).baseAddress!
                try ALEBridge.check(
                    rlx_ale_copy_screen_rgb(handle, ptr, Int32(rgbBuffer.count)),
                    context: "getScreenRGB"
                )
            }
            let floats = rgbBuffer.map { Float($0) }
            return MLXArray(floats, [height, width, 3])
        }
    }
}
