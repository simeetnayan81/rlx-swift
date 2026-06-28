// DiscreteSpace — finite index set {start, start+1, …, start+n-1} (design.md §9.3).

import MLX

/// Finite discrete set of integer indices `start ..< start + n`.
///
/// `Value` is `Int` (scalar). Tensor-valued discrete samples are deferred (vector / PR-05+).
/// `shape` and `dtype` are `nil` because samples are not `MLXArray`.
public struct DiscreteSpace: Space, Equatable, Sendable {
    public typealias Value = Int

    /// Number of valid indices (must be > 0).
    public let n: Int
    /// First valid index (default `0`).
    public let start: Int

    /// - Parameters:
    ///   - n: Count of values; must be positive.
    ///   - start: Lowest valid index (inclusive).
    public init(n: Int, start: Int = 0) {
        precondition(n > 0, "DiscreteSpace.n must be > 0")
        self.n = n
        self.start = start
    }

    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func contains(_ value: Int) -> Bool {
        value >= start && value < start + n
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> Int {
        Int.random(in: start..<(start + n), using: &rng)
    }

    /// MLX-backed sample via `MLXRandom.randInt` on `[start, start+n)`; returns a Swift `Int`.
    public func sample(key: MLXArray) -> Int {
        let upper = start + n
        // Half-open range matches Swift `Int.random(in: start..<(start+n))`.
        let arr = MLXRandom.randInt(Int32(start) ..< Int32(upper), key: key)
        return Device.withDefaultDevice(.cpu) {
            eval(arr)
            return Int(arr.item(Int32.self))
        }
    }
}
