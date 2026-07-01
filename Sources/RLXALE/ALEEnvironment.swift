// ALEEnvironment — Atari via ALE C++ shim (docs/ale-adapter-design.md).
// Dynamics/rendering come from Farama ALE; this type only adapts to Environment.

import Foundation
import MLX
import RLXALECXX
import RLXCore

/// Atari 2600 environment backed by the Arcade Learning Environment C++ library.
///
/// ## Setup
///
/// 1. Install ALE: `./scripts/build-ale.sh ~/.local/ale`
/// 2. Rebuild with `ALE_ROOT` set so the real shim links.
/// 3. Pass any Atari ROM path (e.g. from `ale-py` package roms, or your own `.bin`).
///
/// ## API surface
///
/// - Observation: ``MLXArray`` float32 grayscale `[H,W]` or RGB `[H,W,3]`, values in `0...255`
/// - Action: `Int` index into ALE **minimal** action set
/// - Use with ``OrderEnforcing``, ``TimeLimit``, ``SyncVectorEnv``, etc. like any other env
///
/// ROMs are **not** shipped with rlx-swift.
public final class ALEEnvironment: Environment {
    public typealias Observation = MLXArray
    public typealias Action = Int
    public typealias ObservationSpace = BoxSpace
    public typealias ActionSpace = DiscreteSpace

    public let config: ALEConfig
    public private(set) var observationSpace: BoxSpace
    public private(set) var actionSpace: DiscreteSpace
    public let spec: EnvSpec?

    /// Screen height after ROM load (ALE native resolution).
    public private(set) var screenHeight: Int = 0
    /// Screen width after ROM load.
    public private(set) var screenWidth: Int = 0
    /// Size of the minimal action set for this game.
    public private(set) var minimalActionCount: Int = 0

    private var handle: OpaquePointer?
    private var hasReset = false
    private var episodeOver = false
    private var isClosed = false
    private var lastLives = 0
    private var grayBuffer: [UInt8] = []
    private var rgbBuffer: [UInt8] = []

    /// - Throws: ``EnvironmentError/configuration(_:)`` if ALE is not linked or ROM load fails.
    public init(config: ALEConfig) throws {
        // Prefer CPU for non-app contexts (swift run / Linux) without a Metal metallib.
        Device.setDefault(device: .cpu)

        guard ALEBridge.isLinked() else {
            throw EnvironmentError.configuration(
                "ALE C++ library not linked. Set ALE_ROOT to your ALE install prefix and rebuild (see docs/ale-adapter-design.md)."
            )
        }

        let romURL = URL(fileURLWithPath: config.romPath)
        guard FileManager.default.fileExists(atPath: romURL.path) else {
            throw EnvironmentError.configuration("ROM not found at \(config.romPath)")
        }

        self.config = config
        // Placeholders until ROM metadata is known (BoxSpace needs MLX arrays).
        self.observationSpace = Device.withDefaultDevice(.cpu) {
            BoxSpace(low: 0, high: 255, shape: [1, 1], dtype: .float32)
        }
        self.actionSpace = DiscreteSpace(n: 1)

        let gameName = romURL.deletingPathExtension().lastPathComponent
        self.spec = EnvSpec(
            id: "ALE/\(gameName)-v0",
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
        // Headless by default; set config.displayScreen for SDL window (requires ALE built with SDL).
        _ = rlx_ale_set_bool(raw, "display_screen", config.displayScreen ? 1 : 0)
        _ = rlx_ale_set_bool(raw, "sound", config.sound ? 1 : 0)

        try config.romPath.withCString { cPath in
            try ALEBridge.check(rlx_ale_load_rom(raw, cPath), context: "loadROM \(config.romPath)")
        }

        let h = Int(rlx_ale_screen_height(raw))
        let w = Int(rlx_ale_screen_width(raw))
        let nActions = Int(rlx_ale_minimal_action_count(raw))
        guard h > 0, w > 0, nActions > 0 else {
            rlx_ale_destroy(raw)
            self.handle = nil
            throw EnvironmentError.configuration("ALE ROM loaded but invalid screen/action metadata")
        }
        self.screenHeight = h
        self.screenWidth = w
        self.minimalActionCount = nActions

        switch config.observationType {
        case .grayscale:
            self.observationSpace = Device.withDefaultDevice(.cpu) {
                BoxSpace(low: 0, high: 255, shape: [h, w], dtype: .float32)
            }
            self.grayBuffer = [UInt8](repeating: 0, count: h * w)
        case .rgb:
            self.observationSpace = Device.withDefaultDevice(.cpu) {
                BoxSpace(low: 0, high: 255, shape: [h, w, 3], dtype: .float32)
            }
            self.rgbBuffer = [UInt8](repeating: 0, count: h * w * 3)
        }
        self.actionSpace = DiscreteSpace(n: nActions)
        self.lastLives = Int(rlx_ale_lives(raw))
    }

    /// Convenience: load `directory/<game>.bin` (any Atari game ROM file).
    public convenience init(
        game: String,
        romDirectory: String,
        observationType: ALEObservationType = .grayscale,
        frameSkip: Int = 4,
        livesPolicy: ALELivesPolicy = .gameOverOnly,
        seed: Int32? = 0
    ) throws {
        let path = try ALEConfig.resolveROMPath(game: game, directory: romDirectory)
        try self.init(
            config: ALEConfig(
                romPath: path,
                observationType: observationType,
                frameSkip: frameSkip,
                livesPolicy: livesPolicy,
                seed: seed
            )
        )
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
        info["ale.game_over"] = .bool(gameOver)
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

    /// Copy the current grayscale frame into a Swift array (for demos / PPM export).
    public func copyGrayscaleFrame() throws -> [UInt8] {
        try ensureOpen()
        guard let handle else { throw EnvironmentError.closed }
        var buf = [UInt8](repeating: 0, count: screenHeight * screenWidth)
        try fillGray(handle: handle, buffer: &buf)
        grayBuffer = buf
        return buf
    }

    /// Copy the current RGB frame (interleaved RGB, length `H*W*3`).
    public func copyRGBFrame() throws -> [UInt8] {
        try ensureOpen()
        guard let handle else { throw EnvironmentError.closed }
        var buf = [UInt8](repeating: 0, count: screenHeight * screenWidth * 3)
        try fillRGB(handle: handle, buffer: &buf)
        rgbBuffer = buf
        return buf
    }

    private func ensureOpen() throws {
        if isClosed { throw EnvironmentError.closed }
    }

    private func captureObservation(_ handle: OpaquePointer) throws -> MLXArray {
        try Device.withDefaultDevice(.cpu) {
            switch config.observationType {
            case .grayscale:
                var buf = [UInt8](repeating: 0, count: screenHeight * screenWidth)
                try fillGray(handle: handle, buffer: &buf)
                grayBuffer = buf
                let floats = buf.map { Float($0) }
                return MLXArray(floats, [screenHeight, screenWidth])
            case .rgb:
                var buf = [UInt8](repeating: 0, count: screenHeight * screenWidth * 3)
                try fillRGB(handle: handle, buffer: &buf)
                rgbBuffer = buf
                let floats = buf.map { Float($0) }
                return MLXArray(floats, [screenHeight, screenWidth, 3])
            }
        }
    }

    private func fillGray(handle: OpaquePointer, buffer: inout [UInt8]) throws {
        let n = buffer.count
        try buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else {
                throw EnvironmentError.configuration("gray buffer baseAddress nil")
            }
            try ALEBridge.check(
                rlx_ale_copy_screen_gray(handle, base, Int32(n)),
                context: "getScreenGrayscale"
            )
        }
    }

    private func fillRGB(handle: OpaquePointer, buffer: inout [UInt8]) throws {
        let n = buffer.count
        try buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else {
                throw EnvironmentError.configuration("rgb buffer baseAddress nil")
            }
            try ALEBridge.check(
                rlx_ale_copy_screen_rgb(handle, base, Int32(n)),
                context: "getScreenRGB"
            )
        }
    }
}
