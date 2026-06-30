import RLXCore
import RLXEnvs
import RLXTesting
import RLXWrappers
import XCTest

/// Reset returns observation outside space.
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
        StepResult(observation: 0, reward: 0, terminated: true, truncated: false)
    }

    func close() throws {}
}

final class CheckEnvironmentTests: XCTestCase {

    func testPassesOnDummyEnv() throws {
        try checkEnvironment({ DummyEnv(episodeLength: 5) })
    }

    func testPassesOnOrderEnforcingDummyEnv() throws {
        try checkEnvironment(
            { OrderEnforcing(DummyEnv(episodeLength: 4)) },
            options: CheckEnvironmentOptions(episodes: 3, enforceOrder: false)
        )
    }

    func testEnforceOrderPath() throws {
        try checkEnvironment(
            { DummyEnv(episodeLength: 3) },
            options: CheckEnvironmentOptions(episodes: 2, enforceOrder: true)
        )
    }

    func testFailsOnBadObservation() {
        XCTAssertThrowsError(try checkEnvironment({ BadObsEnv() })) { err in
            guard let checkErr = err as? CheckEnvironmentError,
                  case .observationNotInSpace = checkErr else {
                return XCTFail("expected observationNotInSpace, got \(err)")
            }
        }
    }
}
