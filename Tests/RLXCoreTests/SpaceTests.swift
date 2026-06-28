import MLX
import RLXCore
import XCTest

/// Unit tests for PR-04 `Space`, `DiscreteSpace`, and `BoxSpace`.
///
/// Prefer tier-1: `./scripts/xcodebuild-test.sh`
final class SpaceTests: XCTestCase {

    override func setUp() {
        super.setUp()
        _ = Device.withDefaultDevice(.cpu) { () -> Void in }
    }

    // MARK: - DiscreteSpace

    func testDiscreteMetadataAndContains() {
        let space = DiscreteSpace(n: 3, start: 1)
        XCTAssertNil(space.shape)
        XCTAssertNil(space.dtype)
        XCTAssertTrue(space.contains(1))
        XCTAssertTrue(space.contains(3))
        XCTAssertFalse(space.contains(0))
        XCTAssertFalse(space.contains(4))
        XCTAssertEqual(space, DiscreteSpace(n: 3, start: 1))
        XCTAssertNotEqual(space, DiscreteSpace(n: 3, start: 0))
    }

    func testDiscreteSampleUsingSplitMixReproducible() {
        let space = DiscreteSpace(n: 5)
        var a = SplitMix64(seed: 42)
        var b = SplitMix64(seed: 42)
        for _ in 0..<32 {
            let x = space.sample(using: &a)
            let y = space.sample(using: &b)
            XCTAssertEqual(x, y)
            XCTAssertTrue(space.contains(x))
        }
    }

    func testDiscreteSampleKeyReproducibleAndInRange() {
        Device.withDefaultDevice(.cpu) {
            let space = DiscreteSpace(n: 7, start: 2)
            let key = PRNG.key(from: 99 as UInt64)
            let x = space.sample(key: key)
            let y = space.sample(key: PRNG.key(from: 99 as UInt64))
            XCTAssertEqual(x, y)
            XCTAssertTrue(space.contains(x))
        }
    }

    func testDiscreteSampleKeyIgnoresGlobalMLXSeed() {
        Device.withDefaultDevice(.cpu) {
            let space = DiscreteSpace(n: 4)
            let key = PRNG.key(from: 7 as UInt64)
            let before = space.sample(key: key)
            MLXRandom.seed(0xFFFF_FFFF_FFFF_FFFF)
            let after = space.sample(key: PRNG.key(from: 7 as UInt64))
            XCTAssertEqual(before, after)
        }
    }

    // MARK: - BoxSpace

    func testBoxScalarInitMetadata() {
        let box = BoxSpace(low: -1, high: 1, shape: [2, 3])
        XCTAssertEqual(box.shape, [2, 3])
        XCTAssertEqual(box.dtype, .float32)
        XCTAssertEqual(box, BoxSpace(low: -1, high: 1, shape: [2, 3]))
    }

    func testBoxContainsInAndOutOfBounds() {
        Device.withDefaultDevice(.cpu) {
            let box = BoxSpace(low: 0, high: 1, shape: [2])
            let inside = MLXArray([0.0, 1.0] as [Float])
            let below = MLXArray([-0.1, 0.5] as [Float])
            let above = MLXArray([0.5, 1.1] as [Float])
            let wrongShape = MLXArray([0.5] as [Float])
            eval(inside, below, above, wrongShape)
            XCTAssertTrue(box.contains(inside))
            XCTAssertFalse(box.contains(below))
            XCTAssertFalse(box.contains(above))
            XCTAssertFalse(box.contains(wrongShape))
        }
    }

    func testBoxSampleUsingInBoundsAndReproducible() {
        Device.withDefaultDevice(.cpu) {
            let box = BoxSpace(low: 0, high: 1, shape: [3])
            var a = SplitMix64(seed: 1)
            var b = SplitMix64(seed: 1)
            let x = box.sample(using: &a)
            let y = box.sample(using: &b)
            eval(x, y)
            XCTAssertTrue(PRNG.keysEqual(x, y))
            XCTAssertTrue(box.contains(x))
            XCTAssertEqual(x.shape, [3])
            XCTAssertEqual(x.dtype, .float32)
        }
    }

    func testBoxSampleKeyInBoundsAndReproducible() {
        Device.withDefaultDevice(.cpu) {
            let box = BoxSpace(low: -2, high: 2, shape: [2, 2])
            let k = PRNG.key(from: 123 as UInt64)
            let x = box.sample(key: k)
            let y = box.sample(key: PRNG.key(from: 123 as UInt64))
            eval(x, y)
            XCTAssertTrue(PRNG.keysEqual(x, y))
            XCTAssertEqual(x.shape, [2, 2])
            // Half-open [low, high) — all elements strictly < high and >= low for continuous.
            let vals = x.asArray(Float.self)
            for v in vals {
                XCTAssertGreaterThanOrEqual(v, -2)
                XCTAssertLessThan(v, 2)
            }
        }
    }

    func testBoxSampleKeyIgnoresGlobalMLXSeed() {
        Device.withDefaultDevice(.cpu) {
            let box = BoxSpace(low: 0, high: 1, shape: [2])
            let before = box.sample(key: PRNG.key(from: 5 as UInt64))
            MLXRandom.seed(12345)
            let after = box.sample(key: PRNG.key(from: 5 as UInt64))
            eval(before, after)
            XCTAssertTrue(PRNG.keysEqual(before, after))
        }
    }

    func testBoxUnboundedContainsFiniteSampleRejectedAtSample() {
        Device.withDefaultDevice(.cpu) {
            let low = MLXArray([-Float.infinity] as [Float])
            let high = MLXArray([Float.infinity] as [Float])
            let box = BoxSpace(low: low, high: high)
            let mid = MLXArray([0.0] as [Float])
            eval(mid)
            XCTAssertTrue(box.contains(mid))
        }
    }
}
