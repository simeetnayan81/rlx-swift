// MultiBinarySpace — binary {0,1} tensors (design.md §9.4 / PR-05).

import MLX

/// Binary vector / tensor with entries in `{0, 1}` (`int32`).
public struct MultiBinarySpace: Space, Equatable, @unchecked Sendable {
    public typealias Value = MLXArray

    /// Shape of each sample (e.g. `[8]` for eight bits).
    public let shape: [Int]?
    /// Always `.int32` for v1.
    public let dtype: DType?

    private var resolvedShape: [Int] { shape ?? [] }

    /// Convenience: length-`n` vector.
    public init(n: Int) {
        precondition(n > 0, "MultiBinarySpace.n must be > 0")
        self.shape = [n]
        self.dtype = .int32
    }

    public init(shape: [Int]) {
        precondition(!shape.isEmpty && shape.allSatisfy { $0 > 0 }, "MultiBinarySpace.shape invalid")
        self.shape = shape
        self.dtype = .int32
    }

    public static func == (lhs: MultiBinarySpace, rhs: MultiBinarySpace) -> Bool {
        lhs.resolvedShape == rhs.resolvedShape
    }

    public func contains(_ value: MLXArray) -> Bool {
        guard value.shape == resolvedShape, value.dtype == .int32 else { return false }
        return Device.withDefaultDevice(.cpu) {
            eval(value)
            for x in value.asArray(Int32.self) {
                if x != 0 && x != 1 { return false }
            }
            return true
        }
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> MLXArray {
        let count = resolvedShape.reduce(1, *)
        var bits: [Int32] = []
        bits.reserveCapacity(count)
        for _ in 0..<count {
            bits.append(Int32.random(in: 0...1, using: &rng))
        }
        return MLXArray(bits).reshaped(resolvedShape)
    }

    public func sample(key: MLXArray) -> MLXArray {
        MLXRandom.randInt(
            low: Int32(0),
            high: Int32(2),
            resolvedShape,
            type: Int32.self,
            key: key
        )
    }
}
