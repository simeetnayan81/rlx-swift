// Info — type-erased diagnostic bag for reset/step results (design.md §14.1).

import Foundation
// MLXArray is not Sendable; @preconcurrency keeps Info/InfoValue usable as Sendable
// bags while callers treat array payloads as single-threaded (design.md §14.1).
@preconcurrency import MLX

/// Heterogeneous value stored under an `Info` key.
///
/// Prefer wrapper- or feature-scoped key names (e.g. `TimeLimit.truncated`,
/// nested `episode` metrics). Compile-time key constants land with wrappers
/// (`InfoKeys` in PR-08); core only defines the value model here.
///
/// `@unchecked Sendable` because `.array(MLXArray)` is not formally Sendable;
/// do not share a live `MLXArray` across isolation domains without synchronization.
public enum InfoValue: @unchecked Sendable, Equatable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    /// Nested diagnostic bag (e.g. `episode` metrics from statistics wrappers).
    case nested(Info)
    /// Tensor payload (e.g. vector autoreset `final_observation`).
    ///
    /// `MLXArray` is not formally `Sendable`; equality compares shape, dtype,
    /// and scalar contents after a CPU eval (see `==`). Treat concurrent use
    /// of the same array instance as undefined.
    case array(MLXArray)

    public static func == (lhs: InfoValue, rhs: InfoValue) -> Bool {
        switch (lhs, rhs) {
        case (.bool(let a), .bool(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.double(let a), .double(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.nested(let a), .nested(let b)): return a == b
        case (.array(let a), .array(let b)): return mlxArraysEqual(a, b)
        default: return false
        }
    }
}

/// Type-erased but structured bag for auxiliary environment outputs.
///
/// Value semantics; supports nested `Info` and limited tensor values.
public struct Info: Sendable, Equatable {
    private var storage: [String: InfoValue]

    /// Empty diagnostic bag.
    public init() {
        self.storage = [:]
    }

    /// Build from an explicit key–value dictionary.
    public init(_ values: [String: InfoValue]) {
        self.storage = values
    }

    /// Keys currently present, in arbitrary order.
    public var keys: Dictionary<String, InfoValue>.Keys {
        storage.keys
    }

    /// Whether the bag has no entries.
    public var isEmpty: Bool {
        storage.isEmpty
    }

    /// Number of top-level keys.
    public var count: Int {
        storage.count
    }

    public subscript(key: String) -> InfoValue? {
        get { storage[key] }
        set { storage[key] = newValue }
    }

    /// Insert or replace a value and return `self` for chaining in tests/builders.
    @discardableResult
    public mutating func set(_ key: String, _ value: InfoValue) -> Info {
        storage[key] = value
        return self
    }

    /// Remove a key if present.
    @discardableResult
    public mutating func removeValue(forKey key: String) -> InfoValue? {
        storage.removeValue(forKey: key)
    }

    /// Merge `other` into this bag; keys in `other` overwrite on conflict.
    public mutating func merge(_ other: Info) {
        for (key, value) in other.storage {
            storage[key] = value
        }
    }

    /// Copy with `other` merged on top (non-mutating).
    public func merging(_ other: Info) -> Info {
        var copy = self
        copy.merge(other)
        return copy
    }
}

// MARK: - MLXArray equality helper (InfoValue only)

/// Best-effort structural equality for diagnostic tensors stored in `Info`.
///
/// Compares shape and dtype, then evaluates on CPU and compares scalar buffers
/// for common numeric/bool dtypes. Falls back to `false` for unsupported dtypes
/// rather than claiming equality.
private func mlxArraysEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    if a.shape != b.shape { return false }
    if a.dtype != b.dtype { return false }
    // Zero-size tensors are equal if shape/dtype match.
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
        case .float16:
            return a.asArray(Float16.self) == b.asArray(Float16.self)
        case .bfloat16:
            // No portable scalar buffer API; treat equal only if identical instance.
            return a === b
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
