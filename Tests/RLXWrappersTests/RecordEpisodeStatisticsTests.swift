import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

final class RecordEpisodeStatisticsTests: XCTestCase {

    func testWritesEpisodeMetricsOnDone() throws {
        let env = RecordEpisodeStatistics(DummyEnv(episodeLength: 3))
        _ = try env.reset()
        _ = try env.step(2) // reward 2
        _ = try env.step(1) // reward 1
        let last = try env.step(3) // reward 3; total 6
        XCTAssertTrue(last.done)
        guard case .nested(let episode)? = last.info[InfoKeys.episode] else {
            return XCTFail("expected nested episode info")
        }
        XCTAssertEqual(episode[InfoKeys.episodeReturn], .double(6))
        XCTAssertEqual(episode[InfoKeys.episodeLength], .int(3))
        XCTAssertNil(episode[InfoKeys.episodeTime])
    }

    func testNoEpisodeKeyMidEpisode() throws {
        let env = RecordEpisodeStatistics(DummyEnv(episodeLength: 5))
        _ = try env.reset()
        let mid = try env.step(0)
        XCTAssertFalse(mid.done)
        XCTAssertNil(mid.info[InfoKeys.episode])
    }

    func testResetsAccumulatorsBetweenEpisodes() throws {
        let env = RecordEpisodeStatistics(DummyEnv(episodeLength: 2))
        _ = try env.reset()
        _ = try env.step(1)
        let e1 = try env.step(1)
        guard case .nested(let ep1)? = e1.info[InfoKeys.episode] else {
            return XCTFail("episode 1 metrics")
        }
        XCTAssertEqual(ep1[InfoKeys.episodeReturn], .double(2))
        XCTAssertEqual(ep1[InfoKeys.episodeLength], .int(2))

        _ = try env.reset()
        _ = try env.step(4)
        let e2 = try env.step(0)
        guard case .nested(let ep2)? = e2.info[InfoKeys.episode] else {
            return XCTFail("episode 2 metrics")
        }
        XCTAssertEqual(ep2[InfoKeys.episodeReturn], .double(4))
        XCTAssertEqual(ep2[InfoKeys.episodeLength], .int(2))
    }

    func testRecordTimeOptional() throws {
        let env = RecordEpisodeStatistics(DummyEnv(episodeLength: 1), recordTime: true)
        _ = try env.reset()
        let last = try env.step(0)
        guard case .nested(let episode)? = last.info[InfoKeys.episode] else {
            return XCTFail("expected episode")
        }
        guard case .double(let t)? = episode[InfoKeys.episodeTime] else {
            return XCTFail("expected t")
        }
        XCTAssertGreaterThanOrEqual(t, 0)
    }

    func testComposesWithTimeLimit() throws {
        // Infinite steps truncated by TimeLimit; stats still record on truncated end.
        let env = RecordEpisodeStatistics(
            TimeLimit(DummyEnv(episodeLength: 100), maxEpisodeSteps: 4)
        )
        _ = try env.reset()
        var last: StepResult<Int>!
        for _ in 0..<4 {
            last = try env.step(1)
        }
        XCTAssertTrue(last.truncated)
        XCTAssertFalse(last.terminated)
        guard case .nested(let episode)? = last.info[InfoKeys.episode] else {
            return XCTFail("expected episode on truncation")
        }
        XCTAssertEqual(episode[InfoKeys.episodeReturn], .double(4))
        XCTAssertEqual(episode[InfoKeys.episodeLength], .int(4))
        XCTAssertEqual(last.info[InfoKeys.timeLimitTruncated], .bool(true))
    }

    func testInfoKeysConstants() {
        XCTAssertEqual(InfoKeys.timeLimitTruncated, "TimeLimit.truncated")
        XCTAssertEqual(InfoKeys.episode, "episode")
        XCTAssertEqual(InfoKeys.episodeReturn, "r")
        XCTAssertEqual(InfoKeys.episodeLength, "l")
        XCTAssertEqual(InfoKeys.episodeTime, "t")
    }
}
