// PRNG — thin explicit-key helpers over mlx-swift `MLXRandom` (design.md §8.5, §13.3).
//
// Reuses MLX's PRNG algorithm and `key` / `split` APIs. Does **not** implement a
// second RNG. Library and environment code must never call process-global
// `MLXRandom.seed` / free `seed(_:)` — always thread keys explicitly.

import MLX

/// Explicit MLX PRNG key holder for environment and space randomness.
///
/// Wraps an `MLXArray` key produced by `MLXRandom.key` / `MLXRandom.split`.
/// Copies share the same key tensor identity until advanced; treat as
/// single-threaded like other `MLXArray` payloads (`@unchecked Sendable`).
///
/// ### Global seed policy
/// This type never calls `MLXRandom.seed`. Callers that need draws should pass
/// `key:` into `MLXRandom` sampling functions (or use ``nextKey()`` then sample).
public struct PRNG: @unchecked Sendable {
    /// Current MLX PRNG key (typically shape `[2]`, dtype `uint32`).
    public private(set) var key: MLXArray

    /// Root key from ``Seed`` via `MLXRandom.key` only.
    public init(seed: Seed) {
        self.key = Self.key(from: seed)
    }

    /// Root key from a raw `UInt64` seed.
    public init(seed: UInt64) {
        self.key = Self.key(from: seed)
    }

    /// Resume or adopt an existing MLX key (advanced / tests).
    public init(key: MLXArray) {
        self.key = key
    }

    // MARK: - Root key (MLXRandom.key)

    /// Build a PRNG key from ``Seed`` without touching global MLX RNG state.
    public static func key(from seed: Seed) -> MLXArray {
        MLXRandom.key(seed.rawValue)
    }

    /// Build a PRNG key from `UInt64` without touching global MLX RNG state.
    public static func key(from seed: UInt64) -> MLXArray {
        MLXRandom.key(seed)
    }

    // MARK: - Split (MLXRandom.split)

    /// Split `key` into two independent keys (thin wrapper over `MLXRandom.split`).
    public static func split(_ key: MLXArray, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray) {
        MLXRandom.split(key: key, stream: stream)
    }

    /// Split `key` into `count` independent keys.
    ///
    /// - Precondition: `count >= 1`. For `count == 1`, returns a one-element array
    ///   from `MLXRandom.split(key:into: 1)` (MLX-defined layout).
    public static func split(
        _ key: MLXArray,
        into count: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        precondition(count >= 1, "PRNG.split(into:) requires count >= 1")
        return MLXRandom.split(key: key, into: count, stream: stream)
    }

    // MARK: - In-place advancement

    /// Consume one sub-key for a draw; advance `self.key` to the other half of a 2-way split.
    ///
    /// Equivalent to JAX-style `(key, subkey) = split(key)` where `self` keeps `key`
    /// and the returned value is `subkey` for one `MLXRandom` call.
    public mutating func nextKey(stream: StreamOrDevice = .default) -> MLXArray {
        let (next, consumed) = Self.split(key, stream: stream)
        key = next
        return consumed
    }

    /// Split into `count` keys and replace `self.key` with a fresh key from an extra split.
    ///
    /// Implementation: `split(self.key, into: count + 1)`; last element becomes the new
    /// stored key; first `count` elements are returned. Guarantees the returned keys are
    /// independent of the advanced state.
    public mutating func split(into count: Int, stream: StreamOrDevice = .default) -> [MLXArray] {
        precondition(count >= 1, "PRNG.split(into:) requires count >= 1")
        let parts = Self.split(key, into: count + 1, stream: stream)
        key = parts[count]
        return Array(parts.prefix(count))
    }

    // MARK: - Named env streams (design.md §13.3)

    /// Build the standard three-stream key tree for a single env `reset(seed:)`.
    ///
    /// Order is part of the contract: `MLXRandom.split(key:into: 3)` assigns
    /// `[0] = dynamics`, `[1] = observationNoise`, `[2] = actionNoise`.
    public static func envStreams(from seed: Seed, stream: StreamOrDevice = .default) -> EnvPRNGStreams {
        let root = key(from: seed)
        let parts = split(root, into: 3, stream: stream)
        return EnvPRNGStreams(
            dynamics: parts[0],
            observationNoise: parts[1],
            actionNoise: parts[2]
        )
    }

    /// Same as ``envStreams(from:stream:)`` with a raw `UInt64` seed.
    public static func envStreams(from seed: UInt64, stream: StreamOrDevice = .default) -> EnvPRNGStreams {
        envStreams(from: Seed(seed), stream: stream)
    }

    // MARK: - Key equality (tests / diagnostics)

    /// Structural equality for MLX PRNG keys (shape, dtype, eval'd scalars).
    ///
    /// Evaluates on CPU. Intended for tests and assertions — not a hot path.
    public static func keysEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        prngKeysEqual(a, b)
    }
}

/// Fixed named PRNG streams derived on env `reset(seed:)` (design.md §13.3).
///
/// Envs that only need dynamics may ignore noise streams. Additional streams
/// can be split from these keys without changing the three field names.
public struct EnvPRNGStreams: @unchecked Sendable {
    /// Environment transition stochasticity (primary stream).
    public var dynamics: MLXArray
    /// Observation noise, if any.
    public var observationNoise: MLXArray
    /// Action noise / exploration hooks, if any.
    public var actionNoise: MLXArray

    public init(dynamics: MLXArray, observationNoise: MLXArray, actionNoise: MLXArray) {
        self.dynamics = dynamics
        self.observationNoise = observationNoise
        self.actionNoise = actionNoise
    }
}

// MARK: - Key compare (file-private)

private func prngKeysEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    if a.shape != b.shape { return false }
    if a.dtype != b.dtype { return false }
    let elementCount = a.shape.reduce(1, *)
    if elementCount == 0 { return true }

    return Device.withDefaultDevice(.cpu) {
        eval(a, b)
        switch a.dtype {
        case .bool:
            return a.asArray(Bool.self) == b.asArray(Bool.self)
        case .uint8:
            return a.asArray(UInt8.self) == b.asArray(UInt8.self)
        case .uint16:
            return a.asArray(UInt16.self) == b.asArray(UInt16.self)
        case .uint32:
            return a.asArray(UInt32.self) == b.asArray(UInt32.self)
        case .uint64:
            return a.asArray(UInt64.self) == b.asArray(UInt64.self)
        case .int8:
            return a.asArray(Int8.self) == b.asArray(Int8.self)
        case .int16:
            return a.asArray(Int16.self) == b.asArray(Int16.self)
        case .int32:
            return a.asArray(Int32.self) == b.asArray(Int32.self)
        case .int64:
            return a.asArray(Int64.self) == b.asArray(Int64.self)
        case .float16, .bfloat16:
            let af = a.asType(Float.self)
            let bf = b.asType(Float.self)
            eval(af, bf)
            return af.asArray(Float.self) == bf.asArray(Float.self)
        case .float32:
            return a.asArray(Float.self) == b.asArray(Float.self)
        case .float64:
            return a.asArray(Double.self) == b.asArray(Double.self)
        case .complex64:
            return a === b
        @unknown default:
            return a === b
        }
    }
}
