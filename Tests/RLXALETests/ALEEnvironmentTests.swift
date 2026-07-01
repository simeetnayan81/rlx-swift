import RLXALE
import RLXCore
import XCTest

final class ALEEnvironmentTests: XCTestCase {

    func testReportsLinkStatus() {
        // Without ALE_ROOT this binary uses the stub (false). With ALE_ROOT, true.
        _ = RLXALE.isALELinked
    }

    func testInitThrowsWhenNotLinkedOrMissingROM() throws {
        if !RLXALE.isALELinked {
            XCTAssertThrowsError(
                try ALEEnvironment(config: ALEConfig(romPath: "/nonexistent/rom.bin"))
            ) { err in
                guard case .configuration = err as? EnvironmentError else {
                    return XCTFail("expected configuration, got \(err)")
                }
            }
            return
        }

        // Linked but bad ROM path
        XCTAssertThrowsError(
            try ALEEnvironment(config: ALEConfig(romPath: "/nonexistent/rom.bin"))
        ) { err in
            guard case .configuration = err as? EnvironmentError else {
                return XCTFail("expected configuration, got \(err)")
            }
        }
    }

    func testLiveROMWhenProvided() throws {
        guard RLXALE.isALELinked else {
            throw XCTSkip("ALE not linked (set ALE_ROOT to enable)")
        }
        guard let rom = ProcessInfo.processInfo.environment["ALE_ROM_PATH"], !rom.isEmpty else {
            throw XCTSkip("Set ALE_ROM_PATH to a ROM file to run live ALE tests")
        }

        let env = try ALEEnvironment(
            config: ALEConfig(romPath: rom, observationType: .grayscale, frameSkip: 4, seed: 1)
        )
        let r = try env.reset(seed: 1 as UInt64?, options: nil)
        XCTAssertEqual(r.observation.shape.count, 2)
        let step = try env.step(0)
        XCTAssertTrue(step.reward.isFinite)
        XCTAssertEqual(step.observation.shape, r.observation.shape)
        try env.close()
    }
}
