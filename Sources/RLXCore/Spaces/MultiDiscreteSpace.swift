// MultiDiscreteSpace — product of independent discrete axes (design.md §9.4 / PR-05).

import MLX

/// Product of discrete sets; axis `i` takes values in `0 ..< nvec[i]`.
///
/// `Value` is `[Int]` with `count == nvec.count`. No per-axis `start` in v1 (always 0).
public struct MultiDiscreteSpace: Space, Equatable, Sendable {
    public typealias Value = [Int]

    /// Number of values on each axis; every entry must be > 0.
    public let nvec: [Int]

    public init(nvec: [Int]) {
        precondition(!nvec.isEmpty, "MultiDiscreteSpace.nvec must be non-empty")
        precondition(nvec.allSatisfy { $0 > 0 }, "MultiDiscreteSpace.nvec entries must be > 0")
        self.nvec = nvec
    }

    /// Logical vector length (not an `MLXArray` shape).
    public var shape: [Int]? { [nvec.count] }
    public var dtype: DType? { nil }

    public func contains(_ value: [Int]) -> Bool {
        guard value.count == nvec.count else { return false }
        for i in 0..<nvec.count {
            if value[i] < 0 || value[i] >= nvec[i] { return false }
        }
        return true
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> [Int] {
        nvec.map { n in Int.random(in: 0..<n, using: &rng) }
    }

    public func sample(key: MLXArray) -> [Int] {
        let parts = PRNG.split(key, into: nvec.count)
        var out: [Int] = []
        out.reserveCapacity(nvec.count)
        for i in 0..<nvec.count {
            let n = nvec[i]
            let arr = MLXRandom.randInt(Int32(0) ..< Int32(n), key: parts[i])
            let v = Device.withDefaultDevice(.cpu) { () -> Int in
                eval(arr)
                return Int(arr.item(Int32.self))
            }
            out.append(v)
        }
        return out
    }
}
