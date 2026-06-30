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

final class PendulumTests: XCTestCase {

    func testSpacesAndSpec() {
        let env = PendulumEnv()
        XCTAssertEqual(env.spec?.id, "Pendulum-v1")
        XCTAssertEqual(env.spec?.maxEpisodeSteps, 200)
        XCTAssertEqual(env.observationSpace.shape, [3])
        XCTAssertEqual(env.actionSpace.shape, [1])
    }

    func testResetObsInSpace() throws {
        let env = PendulumEnv()
        let r = try env.reset(seed: UInt64(0), options: nil)
        let o = floats(r.observation)
        XCTAssertEqual(o.count, 3)
        XCTAssertEqual(o[0] * o[0] + o[1] * o[1], 1, accuracy: 1e-5) // cos^2+sin^2
        XCTAssertTrue(env.observationSpace.contains(r.observation))
    }

    func testSeedReproducible() throws {
        let a = PendulumEnv()
        let b = PendulumEnv()
        let ra = try a.reset(seed: UInt64(99), options: nil)
        let rb = try b.reset(seed: UInt64(99), options: nil)
        XCTAssertEqual(floats(ra.observation), floats(rb.observation))
        let act = MLXArray([Float(0.5)])
        let sa = try a.step(act)
        let sb = try b.step(act)
        XCTAssertEqual(floats(sa.observation), floats(sb.observation), accuracy: 1e-5)
        XCTAssertEqual(sa.reward, sb.reward, accuracy: 1e-5)
    }

    func testNeverTerminatesFromDynamics() throws {
        let env = PendulumEnv()
        _ = try env.reset(seed: UInt64(1), options: nil)
        for _ in 0..<50 {
            let s = try env.step(MLXArray([Float(1.5)]))
            XCTAssertFalse(s.terminated)
            XCTAssertFalse(s.truncated)
            XCTAssertLessThanOrEqual(s.reward, 0) // costs are non-negative ⇒ reward ≤ 0
        }
    }

    func testTorqueClipped() throws {
        let env = PendulumEnv()
        _ = try env.reset(seed: UInt64(2), options: nil)
        _ = try env.step(MLXArray([Float(100)]))
        XCTAssertEqual(env.debugLastTorque, 2, accuracy: 1e-5)
        _ = try env.step(MLXArray([Float(-100)]))
        XCTAssertEqual(env.debugLastTorque, -2, accuracy: 1e-5)
    }

    func testAngleNormalize() {
        XCTAssertEqual(PendulumEnv.angleNormalize(0), 0, accuracy: 1e-6)
        // Gymnasium: ((pi + pi) % 2pi) - pi == -pi
        XCTAssertEqual(PendulumEnv.angleNormalize(Float.pi), -Float.pi, accuracy: 1e-5)
        // 3pi/2 -> -pi/2
        XCTAssertEqual(PendulumEnv.angleNormalize(3 * Float.pi / 2), -Float.pi / 2, accuracy: 1e-5)
    }

    func testTimeLimitTruncates() throws {
        let env = TimeLimit(PendulumEnv(), maxEpisodeSteps: 5)
        _ = try env.reset(seed: UInt64(3), options: nil)
        var last: StepResult<MLXArray>!
        for _ in 0..<5 {
            last = try env.step(MLXArray([Float(0)]))
        }
        XCTAssertTrue(last.truncated)
        XCTAssertFalse(last.terminated)
        XCTAssertEqual(last.info[InfoKeys.timeLimitTruncated], .bool(true))
    }

    func testRegistryMake() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerPendulum(on: reg)
        let any = try reg.make("Pendulum-v1")
        XCTAssertEqual(any.spec?.id, "Pendulum-v1")
        let r = try any.reset(seed: Seed(1))
        XCTAssertNotNil(r.observation as? MLXArray)
        let s = try any.step(MLXArray([Float(0)]))
        XCTAssertLessThanOrEqual(s.reward, 0)
        try any.close()
    }

    func testRegisterDefaultsIncludesPendulum() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerDefaults(on: reg)
        XCTAssertTrue(reg.ids.contains("Pendulum-v1"))
        XCTAssertTrue(reg.ids.contains("CartPole-v1"))
        XCTAssertTrue(reg.ids.contains("DummyEnv-v0"))
    }

    func testLifecycle() throws {
        let env = PendulumEnv()
        XCTAssertThrowsError(try env.step(MLXArray([Float(0)]))) { err in
            XCTAssertEqual(err as? EnvironmentError, .notReset)
        }
        _ = try env.reset()
        try env.close()
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }
}

private func XCTAssertEqual(_ a: [Float], _ b: [Float], accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, file: file, line: line)
    for i in a.indices {
        XCTAssertEqual(a[i], b[i], accuracy: accuracy, file: file, line: line)
    }
}
