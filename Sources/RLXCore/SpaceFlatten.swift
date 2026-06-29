// SpaceFlatten — structured / multi spaces → single BoxSpace + unflatten meta (PR-05).

import MLX

/// How one segment of a flat vector maps back to a structured field.
public enum FlattenSegment: Sendable, Equatable {
    /// One-hot block of length `n` (float32) for a discrete axis.
    case discreteOneHot(n: Int)
    /// Contiguous float32 run (box ravel or multi-binary as float).
    case denseFloat(count: Int)
}

/// Metadata to invert flatten.
public struct FlattenMeta: Sendable, Equatable {
    public var segments: [FlattenSegment]
    public var flatShape: [Int]

    public init(segments: [FlattenSegment], flatShape: [Int]) {
        self.segments = segments
        self.flatShape = flatShape
    }

    public var totalDim: Int {
        if let d = flatShape.first { return d }
        return segments.reduce(0) { partial, seg in
            switch seg {
            case .discreteOneHot(let n): return partial + n
            case .denseFloat(let c): return partial + c
            }
        }
    }
}

/// Flatten structured spaces to a single ``BoxSpace`` (float32 vector) and invert samples.
public enum SpaceFlatten {
    public static func box(from space: DiscreteSpace) -> (BoxSpace, FlattenMeta) {
        let n = space.n
        let meta = FlattenMeta(segments: [.discreteOneHot(n: n)], flatShape: [n])
        return (BoxSpace(low: 0, high: 1, shape: [n]), meta)
    }

    public static func box(from space: MultiDiscreteSpace) -> (BoxSpace, FlattenMeta) {
        var segs: [FlattenSegment] = []
        var total = 0
        for n in space.nvec {
            segs.append(.discreteOneHot(n: n))
            total += n
        }
        let meta = FlattenMeta(segments: segs, flatShape: [total])
        return (BoxSpace(low: 0, high: 1, shape: [total]), meta)
    }

    public static func box(from space: MultiBinarySpace) -> (BoxSpace, FlattenMeta) {
        let count = (space.shape ?? []).reduce(1, *)
        let meta = FlattenMeta(segments: [.denseFloat(count: count)], flatShape: [count])
        return (BoxSpace(low: 0, high: 1, shape: [count]), meta)
    }

    public static func box(from space: BoxSpace) -> (BoxSpace, FlattenMeta) {
        let shape = space.shape ?? []
        let count = max(shape.reduce(1, *), 1)
        let meta = FlattenMeta(segments: [.denseFloat(count: count)], flatShape: [count])
        let (minL, maxH) = Device.withDefaultDevice(.cpu) { () -> (Float, Float) in
            let lf = space.low.asType(Float.self)
            let hf = space.high.asType(Float.self)
            eval(lf, hf)
            let la = lf.asArray(Float.self)
            let ha = hf.asArray(Float.self)
            return (la.min() ?? 0, ha.max() ?? 1)
        }
        return (BoxSpace(low: minL, high: maxH, shape: [count]), meta)
    }

    public static func box(from space: DictSpace) -> (BoxSpace, FlattenMeta) {
        compositeBox(children: space.keys.map { space.spaces[$0]! })
    }

    public static func box(from space: TupleSpace) -> (BoxSpace, FlattenMeta) {
        compositeBox(children: space.spaces)
    }

    public static func box(fromErased space: AnySpace) -> (BoxSpace, FlattenMeta) {
        switch space.kind {
        case .discrete(let s): return box(from: s)
        case .box(let s): return box(from: s)
        case .multiDiscrete(let s): return box(from: s)
        case .multiBinary(let s): return box(from: s)
        case .other:
            preconditionFailure("SpaceFlatten: cannot flatten SpaceKind.other")
        }
    }

    private static func compositeBox(children: [AnySpace]) -> (BoxSpace, FlattenMeta) {
        var segs: [FlattenSegment] = []
        var lowScalars: [Float] = []
        var highScalars: [Float] = []
        var total = 0
        for child in children {
            let (cBox, cMeta) = box(fromErased: child)
            segs.append(contentsOf: cMeta.segments)
            let dim = cMeta.totalDim
            total += dim
            let (lo, hi) = boundVectors(cBox, count: dim)
            lowScalars.append(contentsOf: lo)
            highScalars.append(contentsOf: hi)
        }
        let meta = FlattenMeta(segments: segs, flatShape: [total])
        let box = BoxSpace(low: MLXArray(lowScalars), high: MLXArray(highScalars), dtype: .float32)
        return (box, meta)
    }

    // MARK: Flatten values

    public static func flatten(_ value: Int, space: DiscreteSpace) -> MLXArray {
        let n = space.n
        let idx = value - space.start
        precondition((0..<n).contains(idx))
        var oneHot = [Float](repeating: 0, count: n)
        oneHot[idx] = 1
        return MLXArray(oneHot)
    }

    public static func flatten(_ value: [Int], space: MultiDiscreteSpace) -> MLXArray {
        var parts: [Float] = []
        for (i, n) in space.nvec.enumerated() {
            var oneHot = [Float](repeating: 0, count: n)
            let idx = value[i]
            precondition((0..<n).contains(idx))
            oneHot[idx] = 1
            parts.append(contentsOf: oneHot)
        }
        return MLXArray(parts)
    }

    public static func flatten(_ value: MLXArray, space: MultiBinarySpace) -> MLXArray {
        floatRavel(value)
    }

    public static func flatten(_ value: MLXArray, space: BoxSpace) -> MLXArray {
        floatRavel(value)
    }

    public static func flatten(_ value: [String: Any], space: DictSpace) -> MLXArray {
        var parts: [Float] = []
        for k in space.keys {
            parts.append(contentsOf: flattenErased(value: value[k]!, space: space.spaces[k]!))
        }
        return MLXArray(parts)
    }

    public static func flatten(_ value: [Any], space: TupleSpace) -> MLXArray {
        var parts: [Float] = []
        for (i, child) in space.spaces.enumerated() {
            parts.append(contentsOf: flattenErased(value: value[i], space: child))
        }
        return MLXArray(parts)
    }

    public static func flattenErased(value: Any, space: AnySpace) -> [Float] {
        switch space.kind {
        case .discrete(let s):
            guard let v = value as? Int else { return [] }
            return arrayFloats(flatten(v, space: s))
        case .box(let s):
            guard let v = value as? MLXArray else { return [] }
            return arrayFloats(flatten(v, space: s))
        case .multiDiscrete(let s):
            guard let v = value as? [Int] else { return [] }
            return arrayFloats(flatten(v, space: s))
        case .multiBinary(let s):
            guard let v = value as? MLXArray else { return [] }
            return arrayFloats(flatten(v, space: s))
        case .other:
            return []
        }
    }

    // MARK: Unflatten

    public static func unflatten(_ flat: MLXArray, space: DiscreteSpace) -> Int {
        let vals = arrayFloats(flat)
        var argmax = 0
        var best = -Float.infinity
        for (j, x) in vals.enumerated() where x > best {
            best = x
            argmax = j
        }
        return space.start + argmax
    }

    public static func unflatten(_ flat: MLXArray, space: MultiDiscreteSpace) -> [Int] {
        let vals = arrayFloats(flat)
        var out: [Int] = []
        var offset = 0
        for n in space.nvec {
            let slice = vals[offset..<(offset + n)]
            var argmax = 0
            var best = -Float.infinity
            for (j, x) in slice.enumerated() where x > best {
                best = x
                argmax = j
            }
            out.append(argmax)
            offset += n
        }
        return out
    }

    private static func floatRavel(_ value: MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            let f = value.asType(Float.self)
            eval(f)
            return MLXArray(f.asArray(Float.self))
        }
    }

    private static func arrayFloats(_ a: MLXArray) -> [Float] {
        Device.withDefaultDevice(.cpu) {
            let f = a.asType(Float.self)
            eval(f)
            return f.asArray(Float.self)
        }
    }

    private static func boundVectors(_ box: BoxSpace, count: Int) -> ([Float], [Float]) {
        Device.withDefaultDevice(.cpu) {
            let lf = box.low.asType(Float.self)
            let hf = box.high.asType(Float.self)
            eval(lf, hf)
            var lo = lf.asArray(Float.self)
            var hi = hf.asArray(Float.self)
            if lo.count == 1 && count > 1 {
                lo = [Float](repeating: lo[0], count: count)
                hi = [Float](repeating: hi[0], count: count)
            }
            while lo.count < count {
                lo.append(lo.last ?? 0)
                hi.append(hi.last ?? 1)
            }
            return (Array(lo.prefix(count)), Array(hi.prefix(count)))
        }
    }
}
