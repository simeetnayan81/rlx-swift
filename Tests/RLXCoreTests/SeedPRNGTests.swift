import MLX
import RLXCore
import XCTest

/// Unit tests for PR-03 `Seed`, `SplitMix64`, and MLX-backed `PRNG` helpers.
///
/// Prefer `xcodebuild` (Metal shaders) for tier-1:
///   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
///     ./scripts/xcodebuild-test.sh
final class SeedPRNGTests: XCTestCase {

    override func setUp() {
        super.setUp()
        _ = Device.withDefaultDevice(.cpu) { () -> Void in }
    }

    // MARK: - Seed

    func testSeedIdentityAndInterop() {
        let s = Seed(42)
        XCTAssertEqual(s.rawValue, 42)
        XCTAssertEqual(s.uint64, 42)
        XCTAssertEqual(Seed(rawValue: 42), s)
        XCTAssertEqual(s.hashValue, Seed(42).hashValue)
        XCTAssertEqual(s.description, "Seed(42)")
    }

    func testSeedCodableRoundTrip() throws {
        let data = try JSONEncoder().encode(Seed(99))
        let decoded = try JSONDecoder().decode(Seed.self, from: data)
        XCTAssertEqual(decoded, Seed(99))
    }

    // MARK: - SplitMix64

    func testSplitMix64Reproducibility() {
        var a = SplitMix64(seed: 0xDEAD_BEEF)
        var b = SplitMix64(seed: Seed(0xDEAD_BEEF))
        for _ in 0..<32 {
            XCTAssertEqual(a.next(), b.next())
        }
    }

    func testSplitMix64GoldenSequenceDeadBeef() {
        // Locked reference outputs for algorithm stability across releases.
        var rng = SplitMix64(seed: 0xDEAD_BEEF)
        let expected: [UInt64] = [
            0x4adfb90f68c9eb9b,
            0xde586a3141a10922,
            0x021fbc2f8e1cfc1d,
            0x7466ce737be16790,
            0x3bfa8764f685bd1c,
            0xab203e503cb55b3f,
            0x5a2fdc2bf68cedb3,
            0xb30a4ccf430b1b5a,
        ]
        for value in expected {
            XCTAssertEqual(rng.next(), value)
        }
    }

    func testSplitMix64DifferentSeedsDiverge() {
        var a = SplitMix64(seed: 1)
        var b = SplitMix64(seed: 2)
        var differed = false
        for _ in 0..<8 {
            if a.next() != b.next() {
                differed = true
                break
            }
        }
        XCTAssertTrue(differed)
    }

    // MARK: - Seed.child (vector fan-out)

    func testSeedChildDeterminism() {
        let base = Seed(1)
        XCTAssertEqual(base.child(index: 0), base.child(index: 0))
        XCTAssertEqual(base.child(index: 1), base.child(index: UInt64(1)))
        // Golden children for base=1 (locked mix).
        XCTAssertEqual(base.child(index: 0).rawValue, 0x910a2dec89025cc1)
        XCTAssertEqual(base.child(index: 1).rawValue, 0xbeeb8da1658eec67)
        XCTAssertEqual(base.child(index: 2).rawValue, 0xf893a2eefb32555e)
        XCTAssertEqual(base.child(index: 3).rawValue, 0x71c18690ee42c90b)
    }

    func testSeedChildIndependenceAndNotParent() {
        let base = Seed(1)
        var seen = Set<UInt64>()
        for i in 0..<16 {
            let c = base.child(index: i)
            XCTAssertNotEqual(c, base, "child(\(i)) must not equal parent")
            XCTAssertFalse(seen.contains(c.rawValue), "duplicate child at \(i)")
            seen.insert(c.rawValue)
        }
        // Depends only on base and index (recompute).
        XCTAssertEqual(Seed(99).child(index: 3), Seed(99).child(index: 3))
        XCTAssertNotEqual(Seed(99).child(index: 3), Seed(100).child(index: 3))
    }

    // MARK: - PRNG keys (MLX)

    func testPRNGKeyFromSeedReproducible() {
        Device.withDefaultDevice(.cpu) {
            let a = PRNG.key(from: Seed(123))
            let b = PRNG.key(from: 123 as UInt64)
            let c = PRNG(seed: Seed(123)).key
            eval(a, b, c)
            XCTAssertTrue(PRNG.keysEqual(a, b))
            XCTAssertTrue(PRNG.keysEqual(a, c))
            // MLX keys are typically [2] uint32 — assert shape at least non-empty.
            XCTAssertFalse(a.shape.isEmpty || a.shape == [0])
        }
    }

    func testPRNGDifferentSeedsDifferentKeys() {
        Device.withDefaultDevice(.cpu) {
            let a = PRNG.key(from: 1 as UInt64)
            let b = PRNG.key(from: 2 as UInt64)
            eval(a, b)
            XCTAssertFalse(PRNG.keysEqual(a, b))
        }
    }

    func testPRNGSplitTwoWayDeterministic() {
        Device.withDefaultDevice(.cpu) {
            let root = PRNG.key(from: 7 as UInt64)
            let (x0, y0) = PRNG.split(root)
            let (x1, y1) = PRNG.split(root)
            eval(x0, y0, x1, y1)
            XCTAssertTrue(PRNG.keysEqual(x0, x1))
            XCTAssertTrue(PRNG.keysEqual(y0, y1))
            XCTAssertFalse(PRNG.keysEqual(x0, y0))
        }
    }

    func testPRNGSplitIntoCountDeterministic() {
        Device.withDefaultDevice(.cpu) {
            let root = PRNG.key(from: 11 as UInt64)
            let a = PRNG.split(root, into: 4)
            let b = PRNG.split(root, into: 4)
            XCTAssertEqual(a.count, 4)
            XCTAssertEqual(b.count, 4)
            for i in 0..<4 {
                eval(a[i], b[i])
                XCTAssertTrue(PRNG.keysEqual(a[i], b[i]))
            }
            // Pairwise distinct among first split.
            for i in 0..<4 {
                for j in (i + 1)..<4 {
                    XCTAssertFalse(PRNG.keysEqual(a[i], a[j]), "keys \(i) and \(j) should differ")
                }
            }
        }
    }

    func testPRNGNextKeyAdvancesAndReplays() {
        Device.withDefaultDevice(.cpu) {
            var p0 = PRNG(seed: 42)
            var p1 = PRNG(seed: 42)
            let k0a = p0.nextKey()
            let k0b = p1.nextKey()
            let k1a = p0.nextKey()
            let k1b = p1.nextKey()
            eval(k0a, k0b, k1a, k1b)
            XCTAssertTrue(PRNG.keysEqual(k0a, k0b))
            XCTAssertTrue(PRNG.keysEqual(k1a, k1b))
            XCTAssertFalse(PRNG.keysEqual(k0a, k1a))
        }
    }

    func testPRNGInstancesAreIsolated() {
        Device.withDefaultDevice(.cpu) {
            var advanced = PRNG(seed: 5)
            let idle = PRNG(seed: 5)
            _ = advanced.nextKey()
            _ = advanced.nextKey()
            // Idle instance still yields the first subkey of a fresh PRNG(seed: 5).
            var fresh = PRNG(seed: 5)
            let expectedFirst = fresh.nextKey()
            var idleCopy = idle
            let idleFirst = idleCopy.nextKey()
            eval(expectedFirst, idleFirst)
            XCTAssertTrue(PRNG.keysEqual(expectedFirst, idleFirst))
        }
    }

    func testPRNGDoesNotDependOnGlobalMLXSeed() {
        Device.withDefaultDevice(.cpu) {
            let pristine = PRNG.key(from: 12345 as UInt64)
            // Pollute process-global MLX RNG; explicit key(from:) must ignore it.
            MLXRandom.seed(0xFFFF_FFFF_FFFF_FFFF)
            let afterGlobal = PRNG.key(from: 12345 as UInt64)
            eval(pristine, afterGlobal)
            XCTAssertTrue(
                PRNG.keysEqual(pristine, afterGlobal),
                "PRNG.key(from:) must use MLXRandom.key(seed), not global state"
            )
        }
    }

    // MARK: - EnvPRNGStreams

    func testEnvPRNGStreamsReplayAndDistinct() {
        Device.withDefaultDevice(.cpu) {
            let a = PRNG.envStreams(from: Seed(7))
            let b = PRNG.envStreams(from: 7 as UInt64)
            eval(a.dynamics, a.observationNoise, a.actionNoise)
            eval(b.dynamics, b.observationNoise, b.actionNoise)
            XCTAssertTrue(PRNG.keysEqual(a.dynamics, b.dynamics))
            XCTAssertTrue(PRNG.keysEqual(a.observationNoise, b.observationNoise))
            XCTAssertTrue(PRNG.keysEqual(a.actionNoise, b.actionNoise))
            XCTAssertFalse(PRNG.keysEqual(a.dynamics, a.observationNoise))
            XCTAssertFalse(PRNG.keysEqual(a.dynamics, a.actionNoise))
            XCTAssertFalse(PRNG.keysEqual(a.observationNoise, a.actionNoise))
        }
    }

    func testEnvPRNGStreamsOrderMatchesSplitInto3() {
        Device.withDefaultDevice(.cpu) {
            let seed = Seed(99)
            let streams = PRNG.envStreams(from: seed)
            let parts = PRNG.split(PRNG.key(from: seed), into: 3)
            eval(streams.dynamics, parts[0])
            eval(streams.observationNoise, parts[1])
            eval(streams.actionNoise, parts[2])
            XCTAssertTrue(PRNG.keysEqual(streams.dynamics, parts[0]))
            XCTAssertTrue(PRNG.keysEqual(streams.observationNoise, parts[1]))
            XCTAssertTrue(PRNG.keysEqual(streams.actionNoise, parts[2]))
        }
    }

    func testPRNGMutatingSplitIntoLeavesIndependentKeys() {
        Device.withDefaultDevice(.cpu) {
            var prng = PRNG(seed: 3)
            let keys = prng.split(into: 3)
            XCTAssertEqual(keys.count, 3)
            let more = prng.nextKey()
            for k in keys {
                eval(k, more)
                XCTAssertFalse(PRNG.keysEqual(k, more))
            }
        }
    }
}
