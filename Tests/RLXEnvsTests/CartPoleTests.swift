import MLX
import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

private func floats(_ a: MLXArray) -> [Float] {
    Device.withDefaultDevice(.cpu) {
        eval(a)
        return a.asArray(Float.self)
    }
}

final class CartPoleTests: XCTestCase {

    func testSpacesAndSpec() {
        let env = CartPoleEnv()
        XCTAssertEqual(env.spec?.id, "CartPole-v1")
        XCTAssertEqual(env.spec?.maxEpisodeSteps, 500)
        XCTAssertEqual(env.actionSpace.n, 2)
        XCTAssertEqual(env.observationSpace.shape, [4])
    }

    func testResetObservationShapeAndContains() throws {
        let env = CartPoleEnv()
        let r = try env.reset(seed: UInt64(0), options: nil)
        let obs = floats(r.observation)
        XCTAssertEqual(obs.count, 4)
        XCTAssertTrue(env.observationSpace.contains(r.observation))
        // Reset noise in [-0.05, 0.05]
        for v in obs {
            XCTAssertLessThanOrEqual(abs(v), 0.05 + 1e-5)
        }
    }

    func testResetSeedReproducible() throws {
        let a = CartPoleEnv()
        let b = CartPoleEnv()
        let ra = try a.reset(seed: UInt64(123), options: nil)
        let rb = try b.reset(seed: UInt64(123), options: nil)
        XCTAssertEqual(floats(ra.observation), floats(rb.observation))
        let sa = try a.step(0)
        let sb = try b.step(0)
        XCTAssertEqual(floats(sa.observation), floats(sb.observation))
        XCTAssertEqual(sa.reward, sb.reward)
    }

    func testRewardIsOneUntilDone() throws {
        let env = CartPoleEnv()
        _ = try env.reset(seed: UInt64(1), options: nil)
        let s = try env.step(0)
        XCTAssertEqual(s.reward, 1)
        XCTAssertFalse(s.truncated)
    }

    func testInvalidAction() throws {
        let env = CartPoleEnv()
        _ = try env.reset()
        XCTAssertThrowsError(try env.step(2)) { err in
            guard case .invalidAction = err as? EnvironmentError else {
                return XCTFail("\(err)")
            }
        }
    }

    func testLifecycleNotResetAndClosed() throws {
        let env = CartPoleEnv()
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .notReset)
        }
        _ = try env.reset()
        try env.close()
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }

    func testTerminatesPastAngleThreshold() throws {
        // Large initial angle via config noise won't force it; drive pole until fall.
        let env = CartPoleEnv()
        _ = try env.reset(seed: UInt64(0), options: nil)
        var done = false
        var steps = 0
        while !done && steps < 500 {
            // Always push right — pole eventually falls for typical seeds
            let s = try env.step(1)
            done = s.done
            steps += 1
            if s.terminated {
                let obs = floats(s.observation)
                let pastX = abs(obs[0]) > CartPoleEnv.xThreshold
                let pastTheta = abs(obs[2]) > CartPoleEnv.thetaThresholdRadians
                XCTAssertTrue(pastX || pastTheta, "terminated without threshold breach: \(obs)")
                XCTAssertFalse(s.truncated)
            }
        }
        XCTAssertTrue(done, "expected termination within 500 steps")
    }

    func testTimeLimitTruncatesWithoutTerminatingTask() throws {
        let env = TimeLimit(CartPoleEnv(), maxEpisodeSteps: 10)
        _ = try env.reset(seed: UInt64(42), options: nil)
        var last: StepResult<MLXArray>!
        for _ in 0..<10 {
            last = try env.step(0)
        }
        XCTAssertTrue(last.truncated)
        // May or may not have terminated; if only time limit, terminated false is ideal for stable seed
        XCTAssertEqual(last.info[InfoKeys.timeLimitTruncated], .bool(true))
    }

    func testRegistryMakeCartPole() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerCartPole(on: reg)
        let any = try reg.make("CartPole-v1")
        XCTAssertEqual(any.spec?.id, "CartPole-v1")
        let r = try any.reset(seed: Seed(7))
        XCTAssertNotNil(r.observation as? MLXArray)
        let s = try any.step(0)
        XCTAssertEqual(s.reward, 1)
        try any.close()
    }

    func testRegistryRejectsWrongConfigType() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerCartPole(on: reg)
        struct Other: EnvConfig {}
        XCTAssertThrowsError(try reg.make("CartPole-v1", config: Other())) { err in
            guard case .invalidConfig = err as? RegistryError else {
                return XCTFail("\(err)")
            }
        }
    }

    func testRegisterDefaultsIncludesCartPoleAndDummy() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerDefaults(on: reg)
        XCTAssertEqual(reg.ids.sorted(), ["CartPole-v1", "DummyEnv-v0", "Pendulum-v1"])
    }
}
