// TupleSpace — ordered heterogeneous product of spaces (design.md §9.4 / PR-05).

import MLX

/// Fixed-arity product; sample is `[Any]` aligned with ``spaces`` indices.
public struct TupleSpace: Space, @unchecked Sendable {
    public typealias Value = [Any]

    public let spaces: [AnySpace]

    public init(spaces: [AnySpace]) {
        precondition(!spaces.isEmpty, "TupleSpace.spaces must be non-empty")
        self.spaces = spaces
    }

    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func contains(_ value: [Any]) -> Bool {
        guard value.count == spaces.count else { return false }
        for i in 0..<spaces.count {
            if !spaces[i].contains(value[i]) { return false }
        }
        return true
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> [Any] {
        let box = RNGBox(rng)
        return spaces.map { $0.sample(box: box) }
    }

    public func sample(box: RNGBox) -> [Any] {
        spaces.map { $0.sample(box: box) }
    }

    public func sample(key: MLXArray) -> [Any] {
        let parts = PRNG.split(key, into: spaces.count)
        return zip(spaces, parts).map { space, key in space.sample(key: key) }
    }
}
