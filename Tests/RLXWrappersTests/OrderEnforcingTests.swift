import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

/// Inner env that does **not** enforce reset/step order (proves OrderEnforcing alone).
private final class LaxEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 3)
    let actionSpace = DiscreteSpace(n: 3)
    var spec: EnvSpec? { EnvSpec(id: "Lax-v0", maxEpisodeSteps: 2) }

    private var steps = 0
    private var forceTruncate = false

    init(forceTruncate: Bool = false) {
        self.forceTruncate = forceTruncate
    }

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        _ = seed
        _ = options
        steps = 0
        return ResetResult(observation: 0)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        steps += 1
        let done = steps >= 2
        return StepResult(
            observation: action,
            reward: Float(action),
            terminated: done && !forceTruncate,
            truncated: done && forceTruncate
        )
    }

    func close() throws {}
}

final class OrderEnforcingTests: XCTestCase {

    func testStepBeforeResetThrows() {
        let env = OrderEnforcing(LaxEnv())
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .notReset)
        }
    }

    func testStepAfterTerminalThrows() throws {
        let env = OrderEnforcing(LaxEnv())
        _ = try env.reset()
        _ = try env.step(1)
        let last = try env.step(0)
        XCTAssertTrue(last.terminated)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
        _ = try env.reset()
        _ = try env.step(0) // ok again
    }

    func testTruncationAlsoRequiresReset() throws {
        let env = OrderEnforcing(LaxEnv(forceTruncate: true))
        _ = try env.reset()
        _ = try env.step(0)
        let last = try env.step(1)
        XCTAssertTrue(last.truncated)
        XCTAssertTrue(last.done)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
    }

    func testForwardsSpacesSpecAndValues() throws {
        let inner = DummyEnv(observationN: 4, actionN: 3, episodeLength: 2)
        let env = OrderEnforcing(inner)
        XCTAssertEqual(env.observationSpace.n, 4)
        XCTAssertEqual(env.actionSpace.n, 3)
        XCTAssertEqual(env.spec?.id, "DummyEnv-v0")
        let r = try env.reset()
        XCTAssertEqual(r.observation, 0)
        let s = try env.step(2)
        XCTAssertEqual(s.observation, 2)
        XCTAssertEqual(s.reward, 2)
    }

    func testUnwrappedBoxesInner() throws {
        let inner = DummyEnv()
        let env = OrderEnforcing(inner)
        let u = env.unwrapped
        XCTAssertEqual(u.spec?.id, "DummyEnv-v0")
        let r = try u.reset()
        XCTAssertEqual(r.observation as? Int, 0)
    }

    func testCloseForwards() throws {
        let env = OrderEnforcing(DummyEnv())
        _ = try env.reset()
        try env.close()
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }

    func testComposeWithDummyEnv() throws {
        let env = OrderEnforcing(DummyEnv(episodeLength: 2))
        _ = try env.reset(seed: Seed(9))
        _ = try env.step(1)
        let last = try env.step(0)
        XCTAssertTrue(last.terminated)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
    }
}
