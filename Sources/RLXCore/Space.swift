// Space — observation/action set descriptors (design.md §8.1, §9, Appendix A).
//
// Dual sampling:
// - `sample(using:)` — Swift `RandomNumberGenerator` (e.g. SplitMix64); no MLX.
// - `sample(key:)` — MLX-backed via `MLXRandom` with explicit key only; never
//   process-global `MLXRandom.seed`. Do not implement the MLX path with SplitMix64.

import MLX

/// Describes a set of valid observations or actions, with membership and sampling.
///
/// Spaces are pure descriptions: they do **not** own episode state or a long-lived seed.
/// Callers pass an RNG or MLX key on each `sample` call (design.md §9.1).
public protocol Space<Value>: Sendable {
    /// Swift type of a single sample.
    associatedtype Value

    /// Tensor shape when values are array-like; `nil` for non-tensor spaces (e.g. ``DiscreteSpace`` → `Int`).
    var shape: [Int]? { get }

    /// MLX dtype when tensor-backed; `nil` for non-tensor `Value` (e.g. `Int`).
    var dtype: DType? { get }

    /// Membership test; side-effect free and suitable for checkers.
    func contains(_ value: Value) -> Bool

    /// Uniform (or space-defined) sample using a Swift RNG. Must not call `MLXRandom.seed`.
    func sample(using rng: inout some RandomNumberGenerator) -> Value

    /// Sample using an explicit MLX PRNG key (`MLXRandom` only). Must not call `MLXRandom.seed`.
    func sample(key: MLXArray) -> Value
}
