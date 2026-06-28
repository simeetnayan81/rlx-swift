// Seed & SplitMix64 — portable seeding for reset(seed:) and Swift RNG paths
// (design.md §8.5, §13.2–§13.4).
//
// MLX tensor randomness uses explicit keys via `PRNG` / `MLXRandom.key` — never
// process-global `MLXRandom.seed` inside library or environment code.

/// Explicit episode / stream seed for `reset(seed:)` and PRNG construction.
///
/// Wraps `UInt64` for type clarity at call sites. `Environment.reset` (PR-06)
/// remains `seed: UInt64?` in Appendix A; pass `seed.rawValue` or add overloads later.
///
/// Vector sub-env derivation uses ``child(index:)`` (pure integer mix, no MLX)
/// so each slot can call single-env `reset(seed:)` with an independent seed.
public struct Seed: RawRepresentable, Hashable, Codable, Sendable, CustomStringConvertible {
    public var rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    /// Convenience initializer (same as `init(rawValue:)`).
    public init(_ value: UInt64) {
        self.rawValue = value
    }

    /// `UInt64` for APIs that take a plain integer seed (e.g. future `reset`).
    public var uint64: UInt64 { rawValue }

    public var description: String {
        "Seed(\(rawValue))"
    }

    // MARK: - Vector / multi-stream child seeds (design.md §13.4)

    /// Deterministic child seed for sub-environment or logical stream index `index`.
    ///
    /// Uses a fixed 64-bit mix of `rawValue` and `index` (SplitMix64 finalizer over
    /// `base &+ (index &+ 1) &* goldenRatio`). Portable across platforms; does not
    /// depend on MLX backend or global RNG state.
    ///
    /// - Children are **not** required to equal the parent for any index.
    /// - Same `(base, index)` always yields the same child.
    /// - Preferred default for vector env fan-out in PR-13 (`reset(seed: child.rawValue)`
    ///   per slot). MLX `PRNG.split(into:)` remains available for in-env key trees.
    public func child(index: UInt64) -> Seed {
        // 2^64 / φ (golden ratio conjugate), same constant family as SplitMix64.
        let golden: UInt64 = 0x9E37_79B9_7F4A_7C15
        let mixed = rawValue &+ (index &+ 1) &* golden
        return Seed(Self.avalanche64(mixed))
    }

    /// `Int` overload; negative indices are converted with `UInt64(bitPattern:)`.
    public func child(index: Int) -> Seed {
        child(index: UInt64(bitPattern: Int64(index)))
    }

    /// SplitMix64-style finalizer (public only as implementation detail of `child`;
    /// not a general hash API).
    private static func avalanche64(_ z: UInt64) -> UInt64 {
        var x = z
        x = (x ^ (x &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        x = (x ^ (x &>> 27)) &* 0x94D0_49BB_1331_11EB
        return x ^ (x &>> 31)
    }
}

// MARK: - SplitMix64 (normative Swift RNG — design.md §13.3)

/// Deterministic `RandomNumberGenerator` for CPU / non-tensor sampling paths.
///
/// Algorithm: SplitMix64 (Steele, Lea, Flood; widely used for seeding and as a
/// fast stateless mixer). Normative for `Space.sample(using:)` (PR-04) so
/// trajectories stay stable across rlx-swift versions unless intentionally bumped.
///
/// Prefer this over `SystemRandomNumberGenerator` inside environments and spaces
/// when reproducibility matters. For MLX tensor draws, pass explicit keys from
/// ``PRNG`` / `MLXRandom` instead.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public init(seed: Seed) {
        self.state = seed.rawValue
    }

    public mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z &>> 31)
    }
}
