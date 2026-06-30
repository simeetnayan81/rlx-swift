import RLXCore
import RLXEnvs
import XCTest

final class DummyEnvTests: XCTestCase {

    func testSpecIdAndDefaults() {
        let env = DummyEnv()
        XCTAssertEqual(env.spec?.id, "DummyEnv-v0")
        XCTAssertEqual(env.spec?.maxEpisodeSteps, 10)
        XCTAssertEqual(env.spec?.nondeterministic, false)
        XCTAssertEqual(env.episodeLength, 10)
        XCTAssertEqual(env.observationSpace.n, 5)
        XCTAssertEqual(env.actionSpace.n, 5)
    }

    func testLifecycle() throws {
        let env = DummyEnv(episodeLength: 3)
        XCTAssertThrowsError(try env.step(1)) { err in
            XCTAssertEqual(err as? EnvironmentError, .notReset)
        }
        let r0 = try env.reset(seed: UInt64(42), options: nil)
        XCTAssertEqual(r0.observation, 0)
        _ = try env.step(1)
        _ = try env.step(1)
        let last = try env.step(1)
        XCTAssertTrue(last.terminated)
        XCTAssertFalse(last.truncated)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
        _ = try env.reset()
        _ = try env.step(0)
        try env.close()
        try env.close() // idempotent
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }

    func testDynamicsAndReward() throws {
        let env = DummyEnv(observationN: 5, actionN: 5, episodeLength: 4)
        _ = try env.reset()
        let s1 = try env.step(2)
        XCTAssertEqual(s1.observation, 2)
        XCTAssertEqual(s1.reward, 2)
        let s2 = try env.step(3)
        XCTAssertEqual(s2.observation, 0) // (2+3) % 5
        XCTAssertEqual(s2.reward, 3)
    }

    func testInvalidAction() throws {
        let env = DummyEnv(actionN: 2)
        _ = try env.reset()
        XCTAssertThrowsError(try env.step(9)) { err in
            guard case .invalidAction = err as? EnvironmentError else {
                return XCTFail("expected invalidAction, got \(err)")
            }
        }
    }

    func testDeterminismSameSeedAndActions() throws {
        let actions = [1, 2, 0, 1]
        let episodeLength = actions.count
        let a = DummyEnv(episodeLength: episodeLength)
        let b = DummyEnv(episodeLength: episodeLength)
        let ra = try a.reset(seed: UInt64(7), options: nil)
        let rb = try b.reset(seed: UInt64(7), options: nil)
        XCTAssertEqual(ra.observation, rb.observation)
        for action in actions {
            let sa = try a.step(action)
            let sb = try b.step(action)
            XCTAssertEqual(sa.observation, sb.observation)
            XCTAssertEqual(sa.reward, sb.reward)
            XCTAssertEqual(sa.terminated, sb.terminated)
        }
    }

    func testAnyEnvironmentBoxing() throws {
        let any = AnyEnvironment(DummyEnv(episodeLength: 2))
        XCTAssertEqual(any.spec?.id, "DummyEnv-v0")
        let r = try any.reset(seed: Seed(1))
        XCTAssertEqual(r.observation as? Int, 0)
        let s = try any.step(1)
        XCTAssertEqual(s.observation as? Int, 1)
        XCTAssertEqual(s.reward, 1)
        try any.close()
    }

    func testTerminatesExactlyAtEpisodeLength() throws {
        let len = 5
        let env = DummyEnv(episodeLength: len)
        _ = try env.reset()
        for i in 1...len {
            let s = try env.step(0)
            if i < len {
                XCTAssertFalse(s.terminated, "step \(i)")
            } else {
                XCTAssertTrue(s.terminated)
            }
        }
    }
}
