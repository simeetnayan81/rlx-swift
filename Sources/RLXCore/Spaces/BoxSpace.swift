// BoxSpace — axis-aligned tensor region with per-element bounds (design.md §9.2).

import MLX

/// Continuous (or integer) tensor region with elementwise `low` / `high` bounds.
///
/// - `Value` is `MLXArray` with fixed ``shape`` and ``dtype``.
/// - `contains` allows non-finite bounds (`±infinity`) for unbounded axes.
/// - `sample` requires **finite** bounds everywhere (precondition); avoids undefined uniform-on-ℝ.
///
/// MLX path uses `MLXRandom.uniform` / `randInt` with an explicit key only — never SplitMix64
/// or process-global `MLXRandom.seed`.
public struct BoxSpace: Space, @unchecked Sendable {
    public typealias Value = MLXArray

    /// Inclusive lower bound array (broadcast to ``shape`` at construction).
    public let low: MLXArray
    /// Inclusive upper bound for `contains`; sample uses half-open MLX semantics where applicable.
    public let high: MLXArray
    /// Sample / value shape.
    public let shape: [Int]?
    /// Element dtype (default `.float32` at init).
    public let dtype: DType?

    private var resolvedShape: [Int] { shape ?? [] }
    private var resolvedDtype: DType { dtype ?? .float32 }

    /// Build from explicit bound arrays (must match `shape` and each other).
    public init(low: MLXArray, high: MLXArray, dtype: DType? = nil) {
        precondition(low.shape == high.shape, "BoxSpace low and high must have the same shape")
        let dt = dtype ?? low.dtype
        self.low = low
        self.high = high
        self.shape = low.shape
        self.dtype = dt
    }

    /// Scalar bounds broadcast to `shape` with the given `dtype` (default `.float32`).
    public init(
        low: Float,
        high: Float,
        shape: [Int],
        dtype: DType = .float32
    ) {
        // Broadcast scalars to full shape via MLX `full`.
        let lowArr = MLXArray.full(shape, values: MLXArray(low), dtype: dtype)
        let highArr = MLXArray.full(shape, values: MLXArray(high), dtype: dtype)
        self.low = lowArr
        self.high = highArr
        self.shape = shape
        self.dtype = dtype
    }

    /// Element count for Swift-side sampling.
    private var elementCount: Int {
        resolvedShape.reduce(1, *)
    }

    public func contains(_ value: MLXArray) -> Bool {
        guard value.shape == resolvedShape else { return false }
        if value.dtype != resolvedDtype { return false }

        return Device.withDefaultDevice(.cpu) {
            eval(value, low, high)
            switch resolvedDtype {
            case .float32:
                return boxContainsFloat(value.asArray(Float.self), low.asArray(Float.self), high.asArray(Float.self))
            case .float64:
                return boxContainsFloat(value.asArray(Double.self), low.asArray(Double.self), high.asArray(Double.self))
            case .int32:
                return boxContainsInt(value.asArray(Int32.self), low.asArray(Int32.self), high.asArray(Int32.self))
            case .int64:
                return boxContainsInt(value.asArray(Int64.self), low.asArray(Int64.self), high.asArray(Int64.self))
            default:
                // Fallback: promote to float32 for comparison.
                let v = value.asType(Float.self)
                let lo = low.asType(Float.self)
                let hi = high.asType(Float.self)
                eval(v, lo, hi)
                return boxContainsFloat(v.asArray(Float.self), lo.asArray(Float.self), hi.asArray(Float.self))
            }
        }
    }

    public func sample(using rng: inout some RandomNumberGenerator) -> MLXArray {
        preconditionFiniteBoundsForSample()
        let count = elementCount
        // Float32 path is primary; integer dtypes use integer uniform.
        switch resolvedDtype {
        case .int32:
            let lo = Device.withDefaultDevice(.cpu) { () -> [Int32] in
                eval(low)
                return low.asArray(Int32.self)
            }
            let hi = Device.withDefaultDevice(.cpu) { () -> [Int32] in
                eval(high)
                return high.asArray(Int32.self)
            }
            var out = [Int32]()
            out.reserveCapacity(count)
            for i in 0..<count {
                // Inclusive high for contains; sample uses inclusive range when hi > lo.
                let a = lo[i]
                let b = hi[i]
                precondition(a <= b, "BoxSpace sample requires low <= high")
                // Inclusive integer range [a, b] via half-open [a, b+1) when no overflow.
                if a == b {
                    out.append(a)
                } else {
                    out.append(Int32.random(in: a...b, using: &rng))
                }
            }
            return MLXArray(out).reshaped(resolvedShape)
        default:
            let lo = Device.withDefaultDevice(.cpu) { () -> [Float] in
                let f = low.asType(Float.self)
                eval(f)
                return f.asArray(Float.self)
            }
            let hi = Device.withDefaultDevice(.cpu) { () -> [Float] in
                let f = high.asType(Float.self)
                eval(f)
                return f.asArray(Float.self)
            }
            var out = [Float]()
            out.reserveCapacity(count)
            for i in 0..<count {
                let a = lo[i]
                let b = hi[i]
                precondition(a.isFinite && b.isFinite && a <= b, "BoxSpace sample requires finite low <= high")
                if a == b {
                    out.append(a)
                } else {
                    // Inclusive upper via nextDown on exclusive upper when possible.
                    out.append(Float.random(in: a...b, using: &rng))
                }
            }
            let arr = MLXArray(out).reshaped(resolvedShape)
            if resolvedDtype == .float32 {
                return arr
            }
            return arr.asType(resolvedDtype)
        }
    }

    public func sample(key: MLXArray) -> MLXArray {
        preconditionFiniteBoundsForSample()
        switch resolvedDtype {
        case .int32:
            // Half-open [low, high+1) is awkward with arrays; use inclusive via randInt low/high
            // where high is exclusive in MLX — pass high + 1 only for finite ints.
            let hiExclusive = high + MLXArray(Int32(1))
            return MLXRandom.randInt(low: low, high: hiExclusive, resolvedShape, type: Int32.self, key: key)
        default:
            // MLX uniform is half-open [low, high). For inclusive-style boxes with equal bounds,
            // MLX still accepts low == high producing empty interval — use low/high as stored.
            // Prefer closed interpretation for float by sampling [low, high) when low < high;
            // when equal, return low.
            return MLXRandom.uniform(low: low, high: high, resolvedShape, dtype: resolvedDtype, key: key)
        }
    }

    private func preconditionFiniteBoundsForSample() {
        let ok = Device.withDefaultDevice(.cpu) { () -> Bool in
            let lo = low.asType(Float.self)
            let hi = high.asType(Float.self)
            eval(lo, hi)
            let la = lo.asArray(Float.self)
            let ha = hi.asArray(Float.self)
            for i in 0..<la.count {
                if !la[i].isFinite || !ha[i].isFinite { return false }
                if la[i] > ha[i] { return false }
            }
            return true
        }
        precondition(ok, "BoxSpace.sample requires finite low/high with low <= high elementwise")
    }
}

extension BoxSpace: Equatable {
    public static func == (lhs: BoxSpace, rhs: BoxSpace) -> Bool {
        guard lhs.resolvedShape == rhs.resolvedShape else { return false }
        guard lhs.resolvedDtype == rhs.resolvedDtype else { return false }
        return PRNG.keysEqual(lhs.low, rhs.low) && PRNG.keysEqual(lhs.high, rhs.high)
    }
}

// MARK: - Contains helpers

private func boxContainsFloat<T: BinaryFloatingPoint>(_ v: [T], _ lo: [T], _ hi: [T]) -> Bool {
    guard v.count == lo.count, v.count == hi.count else { return false }
    for i in 0..<v.count {
        let x = v[i]
        let a = lo[i]
        let b = hi[i]
        if a.isFinite, x < a { return false }
        if b.isFinite, x > b { return false }
    }
    return true
}

private func boxContainsInt<T: BinaryInteger & Comparable>(_ v: [T], _ lo: [T], _ hi: [T]) -> Bool {
    guard v.count == lo.count, v.count == hi.count else { return false }
    for i in 0..<v.count {
        if v[i] < lo[i] || v[i] > hi[i] { return false }
    }
    return true
}
