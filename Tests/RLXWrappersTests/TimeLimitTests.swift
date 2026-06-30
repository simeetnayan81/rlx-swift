import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

/// Never terminates on its own (for pure truncation tests).
private final class InfiniteEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 2)
    let actionSpace = DiscreteSpace(n: 2)

    private var n = 0

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        _ = seed
        _ = options
        n = 0
        return ResetResult(observation: 0)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        n += 1
        return StepResult(
            observation: action,
            reward: 1,
            terminated: false,
            truncated: false
        )
    }

    func close() throws {}
}

final class TimeLimitTests: XCTestCase {

    func testTruncatesAtMaxWithoutTerminating() throws {
        let env = TimeLimit(InfiniteEnv(), maxEpisodeSteps: 3)
        _ = try env.reset()
        _ = try env.step(0)
        _ = try env.step(0)
        let last = try env.step(0)
        XCTAssertFalse(last.terminated)
        XCTAssertTrue(last.truncated)
        XCTAssertTrue(last.done)
        XCTAssertEqual(last.info[InfoKeys.timeLimitTruncated], .bool(true))
    }

    func testNoTimeLimitKeyBeforeLimit() throws {
        let env = TimeLimit(InfiniteEnv(), maxEpisodeSteps: 5)
        _ = try env.reset()
        let mid = try env.step(1)
        XCTAssertFalse(mid.truncated)
        XCTAssertNil(mid.info[InfoKeys.timeLimitTruncated])
    }

    func testResetsStepCounter() throws {
        let env = TimeLimit(InfiniteEnv(), maxEpisodeSteps: 2)
        _ = try env.reset()
        _ = try env.step(0)
        XCTAssertTrue(try env.step(0).truncated)
        _ = try env.reset()
        let first = try env.step(0)
        XCTAssertFalse(first.truncated)
        XCTAssertNil(first.info[InfoKeys.timeLimitTruncated])
        XCTAssertTrue(try env.step(0).truncated)
    }

    func testDoesNotClearInnerTermination() throws {
        // DummyEnv terminates at episodeLength; wrap with higher limit so inner terminates first.
        let env = TimeLimit(DummyEnv(episodeLength: 2), maxEpisodeSteps: 10)
        _ = try env.reset()
        _ = try env.step(0)
        let last = try env.step(0)
        XCTAssertTrue(last.terminated)
        XCTAssertFalse(last.truncated)
        XCTAssertNil(last.info[InfoKeys.timeLimitTruncated])
    }

    func testBothFlagsWhenLimitMatchesTermination() throws {
        let env = TimeLimit(DummyEnv(episodeLength: 3), maxEpisodeSteps: 3)
        _ = try env.reset()
        _ = try env.step(1)
        _ = try env.step(1)
        let last = try env.step(1)
        XCTAssertTrue(last.terminated)
        XCTAssertTrue(last.truncated)
        XCTAssertEqual(last.info[InfoKeys.timeLimitTruncated], .bool(true))
    }

    func testStackWithOrderEnforcing() throws {
        let env = OrderEnforcing(TimeLimit(InfiniteEnv(), maxEpisodeSteps: 2))
        _ = try env.reset()
        _ = try env.step(0)
        let last = try env.step(0)
        XCTAssertTrue(last.truncated)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
    }

    func testForwardsSpacesAndClose() throws {
        let env = TimeLimit(DummyEnv(observationN: 4, actionN: 3), maxEpisodeSteps: 1)
        XCTAssertEqual(env.observationSpace.n, 4)
        XCTAssertEqual(env.actionSpace.n, 3)
        _ = try env.reset()
        try env.close()
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }
}
