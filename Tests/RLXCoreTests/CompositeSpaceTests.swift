import MLX
import RLXCore
import XCTest

final class CompositeSpaceTests: XCTestCase {
    override func setUp() {
        super.setUp()
        _ = Device.withDefaultDevice(.cpu) { () -> Void in }
    }

    // MARK: MultiDiscrete

    func testMultiDiscreteContainsAndSampleSwift() {
        let space = MultiDiscreteSpace(nvec: [3, 2, 4])
        XCTAssertEqual(space.shape, [3])
        XCTAssertTrue(space.contains([0, 1, 3]))
        XCTAssertFalse(space.contains([0, 1]))
        XCTAssertFalse(space.contains([0, 1, 4]))
        let box = RNGBox(seed: 1)
        var a = RNGBox(seed: 1)
        // Use protocol sample with SplitMix via box for multi draws
        var r1 = SplitMix64(seed: 42)
        var r2 = SplitMix64(seed: 42)
        // Independent seeds for full-sample reproducibility through RNGBox
        let s1 = space.sample(using: &r1)
        let s2 = space.sample(using: &r2)
        // Note: sample(using:) on MultiDiscrete advances SplitMix directly — reproducible
        XCTAssertEqual(s1, s2)
        XCTAssertTrue(space.contains(s1))
        _ = box
        _ = a
    }

    func testMultiDiscreteSampleKeyReproducible() {
        Device.withDefaultDevice(.cpu) {
            let space = MultiDiscreteSpace(nvec: [5, 5])
            let k = PRNG.key(from: 7 as UInt64)
            let a = space.sample(key: k)
            let b = space.sample(key: PRNG.key(from: 7 as UInt64))
            XCTAssertEqual(a, b)
            XCTAssertTrue(space.contains(a))
        }
    }

    // MARK: MultiBinary

    func testMultiBinaryContainsAndSamples() {
        Device.withDefaultDevice(.cpu) {
            let space = MultiBinarySpace(n: 4)
            XCTAssertEqual(space.shape, [4])
            let good = MLXArray([0, 1, 1, 0] as [Int32])
            let bad = MLXArray([0, 1, 2, 0] as [Int32])
            eval(good, bad)
            XCTAssertTrue(space.contains(good))
            XCTAssertFalse(space.contains(bad))
            var rng = SplitMix64(seed: 3)
            let s = space.sample(using: &rng)
            XCTAssertTrue(space.contains(s))
            let k = PRNG.key(from: 3 as UInt64)
            let t = space.sample(key: k)
            let t2 = space.sample(key: PRNG.key(from: 3 as UInt64))
            eval(t, t2)
            XCTAssertTrue(PRNG.keysEqual(t, t2))
            XCTAssertTrue(space.contains(t))
        }
    }

    // MARK: Dict / Tuple

    func testDictSpaceOrderAndContains() {
        let d = DiscreteSpace(n: 3)
        let dict = DictSpace([
            ("a", AnySpace(d)),
            ("b", AnySpace(DiscreteSpace(n: 2))),
        ])
        XCTAssertEqual(dict.keys, ["a", "b"])
        XCTAssertTrue(dict.contains(["a": 1, "b": 0]))
        XCTAssertFalse(dict.contains(["a": 1]))
        XCTAssertFalse(dict.contains(["a": 1, "b": 0, "c": 0]))
        let box = RNGBox(seed: 9)
        let s1 = dict.sample(box: box)
        XCTAssertTrue(dict.contains(s1))
        let k = PRNG.key(from: 9 as UInt64)
        Device.withDefaultDevice(.cpu) {
            let x = dict.sample(key: k)
            let y = dict.sample(key: PRNG.key(from: 9 as UInt64))
            XCTAssertEqual(x["a"] as? Int, y["a"] as? Int)
            XCTAssertEqual(x["b"] as? Int, y["b"] as? Int)
        }
    }

    func testTupleSpace() {
        let tuple = TupleSpace(spaces: [
            AnySpace(DiscreteSpace(n: 2)),
            AnySpace(DiscreteSpace(n: 5)),
        ])
        XCTAssertTrue(tuple.contains([0, 4]))
        XCTAssertFalse(tuple.contains([0]))
        let box = RNGBox(seed: 2)
        let s = tuple.sample(box: box)
        XCTAssertTrue(tuple.contains(s))
    }

    // MARK: Flatten

    func testFlattenMultiDiscreteRoundTrip() {
        let space = MultiDiscreteSpace(nvec: [2, 3])
        let value = [1, 2]
        let flat = SpaceFlatten.flatten(value, space: space)
        Device.withDefaultDevice(.cpu) {
            eval(flat)
            XCTAssertEqual(flat.shape, [5])
        }
        let back = SpaceFlatten.unflatten(flat, space: space)
        XCTAssertEqual(back, value)
        let (box, meta) = SpaceFlatten.box(from: space)
        XCTAssertEqual(meta.totalDim, 5)
        XCTAssertEqual(box.shape, [5])
    }

    func testFlattenDiscreteOneHot() {
        let space = DiscreteSpace(n: 4, start: 1)
        let flat = SpaceFlatten.flatten(3, space: space) // index 2 in 0..<4 relative
        let back = SpaceFlatten.unflatten(flat, space: space)
        XCTAssertEqual(back, 3)
    }

    func testFlattenDictConcatOrder() {
        let dict = DictSpace([
            ("x", AnySpace(DiscreteSpace(n: 2))),
            ("y", AnySpace(DiscreteSpace(n: 3))),
        ])
        let value: [String: Any] = ["x": 1, "y": 2]
        let flat = SpaceFlatten.flatten(value, space: dict)
        Device.withDefaultDevice(.cpu) {
            eval(flat)
            XCTAssertEqual(flat.shape, [5]) // 2 + 3 one-hot
        }
        let (box, meta) = SpaceFlatten.box(from: dict)
        XCTAssertEqual(meta.totalDim, 5)
        XCTAssertEqual(box.shape, [5])
    }

    func testNoGlobalSeedOnMultiDiscreteKey() {
        Device.withDefaultDevice(.cpu) {
            let space = MultiDiscreteSpace(nvec: [4])
            let before = space.sample(key: PRNG.key(from: 1 as UInt64))
            MLXRandom.seed(999)
            let after = space.sample(key: PRNG.key(from: 1 as UInt64))
            XCTAssertEqual(before, after)
        }
    }
}
