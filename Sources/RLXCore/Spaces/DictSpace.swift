// DictSpace — named heterogeneous product of spaces (design.md §9.4 / PR-05).

import MLX

/// Structured space: each key maps to an ``AnySpace`` child.
///
/// ``keys`` is the **canonical order** for sampling, contains checks, and flatten (PR-05).
public struct DictSpace: Space, @unchecked Sendable {
    public typealias Value = [String: Any]

    /// Canonical key order (authoritative).
    public let keys: [String]
    /// Child spaces keyed by name; must contain every entry of ``keys``.
    public let spaces: [String: AnySpace]

    /// - Parameters:
    ///   - keys: Visit order for sample / flatten.
    ///   - spaces: Children; must include all `keys` and no requirement on extras (extras ignored).
    public init(keys: [String], spaces: [String: AnySpace]) {
        precondition(!keys.isEmpty, "DictSpace.keys must be non-empty")
        precondition(Set(keys).count == keys.count, "DictSpace.keys must be unique")
        for k in keys {
            precondition(spaces[k] != nil, "DictSpace missing space for key \(k)")
        }
        self.keys = keys
        self.spaces = spaces
    }

    /// Build from erased pairs; key order is the array order.
    public init(_ ordered: [(String, AnySpace)]) {
        precondition(!ordered.isEmpty)
        let keys = ordered.map(\.0)
        var map: [String: AnySpace] = [:]
        for (k, s) in ordered {
            precondition(map[k] == nil, "duplicate key \(k)")
            map[k] = s
        }
        self.init(keys: keys, spaces: map)
    }

    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func contains(_ value: [String: Any]) -> Bool {
        guard value.count == keys.count else { return false }
        for k in keys {
            guard let v = value[k], let child = spaces[k] else { return false }
            if !child.contains(v) { return false }
        }
        for k in value.keys where !keys.contains(k) {
            return false
        }
        return true
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> [String: Any] {
        let box = RNGBox(rng)
        var out: [String: Any] = [:]
        out.reserveCapacity(keys.count)
        for k in keys {
            out[k] = spaces[k]!.sample(box: box)
        }
        return out
    }

    /// Shared-stream sampling for tests (one ``RNGBox`` for all keys).
    public func sample(box: RNGBox) -> [String: Any] {
        var out: [String: Any] = [:]
        for k in keys {
            out[k] = spaces[k]!.sample(box: box)
        }
        return out
    }

    public func sample(key: MLXArray) -> [String: Any] {
        let parts = PRNG.split(key, into: keys.count)
        var out: [String: Any] = [:]
        for (i, k) in keys.enumerated() {
            out[k] = spaces[k]!.sample(key: parts[i])
        }
        return out
    }
}
