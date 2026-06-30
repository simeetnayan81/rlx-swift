import RLXCore
import RLXEnvs
import RLXVector
import RLXWrappers
import XCTest

final class AsyncVectorEnvTests: XCTestCase {

    func testResetSeedsAndBatchShape() async throws {
        let vec = AsyncVectorEnv(numEnvs: 3, autoresetMode: .disabled, maxConcurrency: 2) {
            AnyEnvironment(DummyEnv(episodeLength: 5))
        }
        let r = try await vec.reset(seed: 42)
        XCTAssertEqual(r.observations.count, 3)
        XCTAssertEqual(r.infos.count, 3)
        let s = try await vec.step([0, 1, 2])
        XCTAssertEqual(s.rewards, [0, 1, 2])
        XCTAssertEqual(s.observations[0] as? Int, 0)
        XCTAssertEqual(s.observations[1] as? Int, 1)
        XCTAssertEqual(s.observations[2] as? Int, 2)
        try await vec.close()
    }

    func testStepResultsPreserveIndexOrder() async throws {
        // Even with full concurrency, rewards/obs must match action index order.
        // DummyEnv default actionN=5 → actions in 0..<5 only.
        let vec = AsyncVectorEnv(numEnvs: 8, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(observationN: 10, actionN: 10, episodeLength: 20))
        }
        _ = try await vec.reset()
        let actions = (0..<8).map { $0 as Any }
        let s = try await vec.step(actions)
        for i in 0..<8 {
            XCTAssertEqual(s.rewards[i], Float(i))
            XCTAssertEqual(s.observations[i] as? Int, i)
        }
        try await vec.close()
    }

    func testMaxConcurrencyOneSerialMode() async throws {
        let vec = AsyncVectorEnv(numEnvs: 4, autoresetMode: .disabled, maxConcurrency: 1) {
            AnyEnvironment(DummyEnv(episodeLength: 10))
        }
        let limit = await vec.maxConcurrency
        XCTAssertEqual(limit, 1)
        _ = try await vec.reset(seed: 1)
        let s = try await vec.step([1, 2, 3, 4])
        XCTAssertEqual(s.rewards, [1, 2, 3, 4])
        try await vec.close()
    }

    func testSameStepAutoresetStashesFinalObservation() async throws {
        let vec = AsyncVectorEnv(numEnvs: 1, autoresetMode: .sameStep) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try await vec.reset()
        let s = try await vec.step([1])
        XCTAssertTrue(s.terminateds[0])
        XCTAssertEqual(s.observations[0] as? Int, 0)
        XCTAssertEqual(s.infos[0][InfoKeys.finalObservation], .int(1))
        guard case .nested = s.infos[0][InfoKeys.finalInfo] else {
            return XCTFail("expected final_info nested")
        }
        let s2 = try await vec.step([2])
        XCTAssertEqual(s2.observations[0] as? Int, 0)
        XCTAssertEqual(s2.infos[0][InfoKeys.finalObservation], .int(2))
        try await vec.close()
    }

    func testNextStepAutoreset() async throws {
        let vec = AsyncVectorEnv(numEnvs: 1, autoresetMode: .nextStep) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try await vec.reset()
        let s1 = try await vec.step([3])
        XCTAssertTrue(s1.terminateds[0])
        XCTAssertEqual(s1.observations[0] as? Int, 3)
        XCTAssertNil(s1.infos[0][InfoKeys.finalObservation])
        let s2 = try await vec.step([1])
        XCTAssertEqual(s2.observations[0] as? Int, 1)
        try await vec.close()
    }

    func testDisabledAutoresetThrowsOnSecondStepAfterDone() async throws {
        let vec = AsyncVectorEnv(numEnvs: 1, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(episodeLength: 1))
        }
        _ = try await vec.reset()
        _ = try await vec.step([0])
        do {
            _ = try await vec.step([0])
            XCTFail("expected episodeEnded")
        } catch let err as EnvironmentError {
            XCTAssertEqual(err, .episodeEnded)
        }
        try await vec.close()
    }

    func testBatchSizeMismatch() async throws {
        let vec = AsyncVectorEnv(numEnvs: 2, autoresetMode: .disabled) {
            AnyEnvironment(DummyEnv(episodeLength: 5))
        }
        _ = try await vec.reset()
        do {
            _ = try await vec.step([0])
            XCTFail("expected batchSizeMismatch")
        } catch let err as VectorEnvironmentError {
            XCTAssertEqual(err, .batchSizeMismatch(expected: 2, actual: 1))
        }
        try await vec.close()
    }

    func testCloseIsIdempotentAndBlocksFurtherSteps() async throws {
        let vec = AsyncVectorEnv(numEnvs: 2, autoresetMode: .sameStep) {
            AnyEnvironment(DummyEnv(episodeLength: 5))
        }
        _ = try await vec.reset()
        try await vec.close()
        try await vec.close()
        do {
            _ = try await vec.step([0, 0])
            XCTFail("expected closed")
        } catch let err as VectorEnvironmentError {
            XCTAssertEqual(err, .closed)
        }
    }

    func testTimeLimitTruncationWithSameStep() async throws {
        let vec = AsyncVectorEnv(numEnvs: 2, autoresetMode: .sameStep) {
            AnyEnvironment(TimeLimit(DummyEnv(episodeLength: 100), maxEpisodeSteps: 2))
        }
        _ = try await vec.reset()
        _ = try await vec.step([0, 0])
        let s = try await vec.step([0, 0])
        XCTAssertTrue(s.truncateds[0] && s.truncateds[1])
        XCTAssertEqual(s.observations[0] as? Int, 0)
        XCTAssertEqual(s.infos[0][InfoKeys.timeLimitTruncated], .bool(true))
        try await vec.close()
    }

    func testIndexedFactory() async throws {
        let vec = AsyncVectorEnv(numEnvs: 3, autoresetMode: .disabled) { (i: Int) in
            AnyEnvironment(DummyEnv(episodeLength: 10 + i))
        }
        _ = try await vec.reset(seed: 7)
        let s = try await vec.step([1, 1, 1])
        XCTAssertEqual(s.rewards, [1, 1, 1])
        try await vec.close()
    }

    func testCloseCancelsOutstandingStep() async throws {
        let vec = AsyncVectorEnv(numEnvs: 4, autoresetMode: .disabled, maxConcurrency: 4) {
            AnyEnvironment(DummyEnv(episodeLength: 50))
        }
        _ = try await vec.reset()
        async let stepResult: Result<VectorStepResult, Error> = {
            do {
                return .success(try await vec.step([0, 0, 0, 0]))
            } catch {
                return .failure(error)
            }
        }()
        // Race close against the in-flight batch; at least one path must complete cleanly.
        try await vec.close()
        let outcome = await stepResult
        switch outcome {
        case .success:
            break // step finished before cancel took effect
        case .failure(let err):
            XCTAssertTrue(
                (err as? VectorEnvironmentError) == .cancelled
                    || (err as? VectorEnvironmentError) == .closed,
                "unexpected error: \(err)"
            )
        }
        // Further use is closed
        do {
            _ = try await vec.reset()
            XCTFail("expected closed after close()")
        } catch let err as VectorEnvironmentError {
            XCTAssertEqual(err, .closed)
        }
    }
}
