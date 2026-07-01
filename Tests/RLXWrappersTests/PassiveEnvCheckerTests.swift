import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

/// Returns observation 99 (outside Discrete(2)).
private final class BadObsEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 2)
    let actionSpace = DiscreteSpace(n: 2)

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        ResetResult(observation: 99)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        StepResult(observation: 99, reward: 0, terminated: true, truncated: false)
    }

    func close() throws {}
}

/// Returns NaN reward.
private final class NaNRewardEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 2)
    let actionSpace = DiscreteSpace(n: 2)
    private var hasReset = false

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        hasReset = true
        return ResetResult(observation: 0)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        _ = hasReset
        return StepResult(observation: 0, reward: .nan, terminated: true, truncated: false)
    }

    func close() throws {}
}

final class PassiveEnvCheckerTests: XCTestCase {

    func testForwardsValidDummyEnv() throws {
        let env = PassiveEnvChecker(DummyEnv(episodeLength: 3))
        let r = try env.reset(seed: 1 as UInt64?, options: nil)
        XCTAssertEqual(r.observation, 0)
        let s = try env.step(1)
        XCTAssertEqual(s.reward, 1)
        XCTAssertEqual(s.observation, 1)
        try env.close()
    }

    func testRejectsResetObservationOutsideSpace() {
        let env = PassiveEnvChecker(BadObsEnv())
        XCTAssertThrowsError(try env.reset()) { err in
            guard case .invalidObservation = err as? EnvironmentError else {
                return XCTFail("expected invalidObservation, got \(err)")
            }
        }
    }

    func testRejectsStepObservationOutsideSpace() throws {
        // After a bad reset we never get here; use a env that resets OK but steps badly.
        let env = PassiveEnvChecker(BadStepObsEnv())
        _ = try env.reset()
        XCTAssertThrowsError(try env.step(0)) { err in
            guard case .invalidObservation = err as? EnvironmentError else {
                return XCTFail("expected invalidObservation, got \(err)")
            }
        }
    }

    func testRejectsActionOutsideSpace() throws {
        let env = PassiveEnvChecker(DummyEnv(observationN: 5, actionN: 2, episodeLength: 5))
        _ = try env.reset()
        XCTAssertThrowsError(try env.step(9)) { err in
            guard case .invalidAction = err as? EnvironmentError else {
                return XCTFail("expected invalidAction, got \(err)")
            }
        }
    }

    func testRejectsNonFiniteReward() throws {
        let env = PassiveEnvChecker(NaNRewardEnv())
        _ = try env.reset()
        XCTAssertThrowsError(try env.step(0)) { err in
            guard case .configuration = err as? EnvironmentError else {
                return XCTFail("expected configuration, got \(err)")
            }
        }
    }

    func testComposesWithOrderEnforcingAndTimeLimit() throws {
        let env = PassiveEnvChecker(
            OrderEnforcing(
                TimeLimit(DummyEnv(episodeLength: 100), maxEpisodeSteps: 2)
            )
        )
        _ = try env.reset()
        _ = try env.step(0)
        let last = try env.step(0)
        XCTAssertTrue(last.truncated)
        try env.close()
    }

    func testSpacesAndSpecForwarded() throws {
        let inner = DummyEnv(episodeLength: 2)
        let env = PassiveEnvChecker(inner)
        XCTAssertEqual(env.observationSpace.n, inner.observationSpace.n)
        XCTAssertEqual(env.actionSpace.n, inner.actionSpace.n)
        XCTAssertEqual(env.spec?.id, inner.spec?.id)
        _ = env.unwrapped
        try env.close()
    }
}

/// Reset OK; step returns OOB observation.
private final class BadStepObsEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 2)
    let actionSpace = DiscreteSpace(n: 2)

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        ResetResult(observation: 0)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        StepResult(observation: 99, reward: 0, terminated: true, truncated: false)
    }

    func close() throws {}
}
