// AnySpace — type eraser for heterogeneous composite spaces (PR-05; reused by PR-06).

import MLX

/// Kind tag so ``SpaceFlatten`` can encode erased children without recovering generics.
public enum SpaceKind: Sendable, Equatable {
    case discrete(DiscreteSpace)
    case box(BoxSpace)
    case multiDiscrete(MultiDiscreteSpace)
    case multiBinary(MultiBinarySpace)
    /// Nested composites are not flattened via erasure in v1 (use typed APIs).
    case other
}

/// Type-erased ``Space`` so ``DictSpace`` / ``TupleSpace`` can hold mixed children.
///
/// Values are passed as `Any`; wrong dynamic types yield `contains == false`.
///
/// Sampling through ``sample(box:)`` shares one ``RNGBox`` stream across children.
public final class AnySpace: @unchecked Sendable {
    public let kind: SpaceKind
    private let _shape: [Int]?
    private let _dtype: DType?
    private let _contains: (Any) -> Bool
    private let _sampleBox: (RNGBox) -> Any
    private let _sampleKey: (MLXArray) -> Any

    public init(_ space: DiscreteSpace) {
        self.kind = .discrete(space)
        self._shape = space.shape
        self._dtype = space.dtype
        let captured = space
        self._contains = { ($0 as? Int).map(captured.contains) ?? false }
        self._sampleBox = { box in
            var rng = box
            return captured.sample(using: &rng)
        }
        self._sampleKey = { captured.sample(key: $0) }
    }

    public init(_ space: BoxSpace) {
        self.kind = .box(space)
        self._shape = space.shape
        self._dtype = space.dtype
        let captured = space
        self._contains = { ($0 as? MLXArray).map(captured.contains) ?? false }
        self._sampleBox = { box in
            var rng = box
            return captured.sample(using: &rng)
        }
        self._sampleKey = { captured.sample(key: $0) }
    }

    public init(_ space: MultiDiscreteSpace) {
        self.kind = .multiDiscrete(space)
        self._shape = space.shape
        self._dtype = space.dtype
        let captured = space
        self._contains = { ($0 as? [Int]).map(captured.contains) ?? false }
        self._sampleBox = { box in
            var rng = box
            return captured.sample(using: &rng)
        }
        self._sampleKey = { captured.sample(key: $0) }
    }

    public init(_ space: MultiBinarySpace) {
        self.kind = .multiBinary(space)
        self._shape = space.shape
        self._dtype = space.dtype
        let captured = space
        self._contains = { ($0 as? MLXArray).map(captured.contains) ?? false }
        self._sampleBox = { box in
            var rng = box
            return captured.sample(using: &rng)
        }
        self._sampleKey = { captured.sample(key: $0) }
    }

    /// Generic fallback (no flatten kind — ``SpaceKind/other``).
    public init<S: Space>(other space: S) {
        self.kind = .other
        self._shape = space.shape
        self._dtype = space.dtype
        let captured = space
        self._contains = { value in
            guard let typed = value as? S.Value else { return false }
            return captured.contains(typed)
        }
        self._sampleBox = { box in
            var rng = box
            return captured.sample(using: &rng)
        }
        self._sampleKey = { captured.sample(key: $0) }
    }

    public var shape: [Int]? { _shape }
    public var dtype: DType? { _dtype }

    public func contains(_ value: Any) -> Bool { _contains(value) }

    public func sample(box: RNGBox) -> Any { _sampleBox(box) }

    public func sample(using rng: inout some RandomNumberGenerator) -> Any {
        let box = RNGBox(rng)
        return _sampleBox(box)
    }

    public func sample(key: MLXArray) -> Any { _sampleKey(key) }
}

/// Class-backed ``RandomNumberGenerator`` for multi-child sampling with one shared stream.
public final class RNGBox: RandomNumberGenerator, @unchecked Sendable {
    private var base: any RandomNumberGenerator

    public init(_ rng: some RandomNumberGenerator) {
        self.base = rng
    }

    public init(seed: UInt64) {
        self.base = SplitMix64(seed: seed)
    }

    public func next() -> UInt64 {
        base.next()
    }
}
