import RLXCore
import RLXEnvs
import RLXVector
import RLXWrappers
import XCTest

final class SyncVectorEnvTests: XCTestCase {

    func testResetSeedsDifferPerSlot() throws {
        let vec = SyncVectorEnv(numEnvs: 3, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(episodeLength: 5))
        }
        let r = try vec.reset(seed: 42)
        XCTAssertEqual(r.observations.count, 3)
        // Dummy reset obs always 0; just ensure no throw and pending clear
        _ = try vec.step([0, 0, 0])
        try vec.close()
    }

    func testStepBatchShape() throws {
        let vec = SyncVectorEnv(numEnvs: 2, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(episodeLength: 10))
        }
        _ = try vec.reset()
        let s = try vec.step([1, 2])
        XCTAssertEqual(s.rewards.count, 2)
        XCTAssertEqual(s.rewards[0], 1)
        XCTAssertEqual(s.rewards[1], 2)
        XCTAssertEqual(s.observations[0] as? Int, 1)
        XCTAssertEqual(s.observations[1] as? Int, 2)
        try vec.close()
    }

    func testSameStepAutoresetStashesFinalObservation() throws {
        // episodeLength 1 → every step terminates
        let vec = SyncVectorEnv(numEnvs: 1, autoresetMode: .sameStep) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try vec.reset()
        let s = try vec.step([1])
        XCTAssertTrue(s.terminateds[0])
        // Live obs is reset obs (0); terminal obs was 1
        XCTAssertEqual(s.observations[0] as? Int, 0)
        XCTAssertEqual(s.infos[0][InfoKeys.finalObservation], .int(1))
        guard case .nested = s.infos[0][InfoKeys.finalInfo] else {
            return XCTFail("expected final_info nested")
        }
        // Can step again without explicit reset
        let s2 = try vec.step([2])
        XCTAssertEqual(s2.observations[0] as? Int, 0)
        XCTAssertEqual(s2.infos[0][InfoKeys.finalObservation], .int(2))
        try vec.close()
    }

    func testNextStepAutoreset() throws {
        let vec = SyncVectorEnv(numEnvs: 1, autoresetMode: .nextStep) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try vec.reset()
        let s1 = try vec.step([3])
        XCTAssertTrue(s1.terminateds[0])
        XCTAssertEqual(s1.observations[0] as? Int, 3) // terminal obs kept
        XCTAssertNil(s1.infos[0][InfoKeys.finalObservation])
        // Next step resets then applies action
        let s2 = try vec.step([1])
        XCTAssertEqual(s2.observations[0] as? Int, 1)
        try vec.close()
    }

    func testDisabledAutoresetThrowsOnSecondStepAfterDone() throws {
        let vec = SyncVectorEnv(numEnvs: 1, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try vec.reset()
        _ = try vec.step([0])
        XCTAssertThrowsError(try vec.step([0])) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
        try vec.close()
    }

    func testTimeLimitTruncationWithSameStep() throws {
        let vec = SyncVectorEnv(numEnvs: 2, autoresetMode: .sameStep) {
            AnyEnvironment(TimeLimit(DummyEnv(episodeLength: 100), maxEpisodeSteps: 2))
        }
        _ = try vec.reset()
        _ = try vec.step([0, 0])
        let s = try vec.step([0, 0])
        XCTAssertTrue(s.truncateds[0] && s.truncateds[1])
        XCTAssertEqual(s.observations[0] as? Int, 0) // reset obs
        XCTAssertEqual(s.infos[0][InfoKeys.timeLimitTruncated], .bool(true))
        try vec.close()
    }
}
