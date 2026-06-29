import RLXCore
import XCTest

/// Minimal env for PR-06 protocol / erasure tests (not a shipped product; DummyEnv is PR-07).
private final class CounterEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 5)
    let actionSpace = DiscreteSpace(n: 5)
    let spec: EnvSpec? = EnvSpec(id: "Counter-v0", maxEpisodeSteps: 3, version: 1)

    private var obs: Int = 0
    private var stepsInEpisode: Int = 0
    private var hasReset = false
    private var episodeOver = false
    private var isClosed = false
    private var lastSeed: UInt64?

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        if isClosed { throw EnvironmentError.closed }
        if let seed { lastSeed = seed }
        hasReset = true
        episodeOver = false
        stepsInEpisode = 0
        obs = 0
        return ResetResult(observation: obs)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        if isClosed { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        if episodeOver { throw EnvironmentError.episodeEnded }
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction("action \(action) out of space")
        }
        obs = (obs + action) % 5
        stepsInEpisode += 1
        let terminated = stepsInEpisode >= 3
        if terminated { episodeOver = true }
        return StepResult(
            observation: obs,
            reward: Float(action),
            terminated: terminated,
            truncated: false
        )
    }

    func close() throws {
        isClosed = true
    }

    var debugLastSeed: UInt64? { lastSeed }
}

final class EnvironmentProtocolTests: XCTestCase {

    func testEnvSpecFields() {
        let s = EnvSpec(
            id: "CartPole-v1",
            maxEpisodeSteps: 500,
            rewardThreshold: 475,
            nondeterministic: false,
            defaultRenderMode: RenderMode.none,
            version: 1
        )
        XCTAssertEqual(s.id, "CartPole-v1")
        XCTAssertEqual(s.maxEpisodeSteps, 500)
        XCTAssertEqual(s.defaultRenderMode, RenderMode.none)
        XCTAssertEqual(
            s,
            EnvSpec(
                id: "CartPole-v1",
                maxEpisodeSteps: 500,
                rewardThreshold: 475,
                defaultRenderMode: RenderMode.none,
                version: 1
            )
        )
    }

    func testCounterEnvLifecycle() throws {
        let env = CounterEnv()
        XCTAssertThrowsError(try env.step(1)) { err in
            XCTAssertEqual(err as? EnvironmentError, .notReset)
        }
        let r0 = try env.reset(seed: UInt64(42), options: nil)
        XCTAssertEqual(r0.observation, 0)
        XCTAssertEqual(env.debugLastSeed, 42)
        _ = try env.step(1)
        _ = try env.step(1)
        let last = try env.step(1)
        XCTAssertTrue(last.terminated)
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .episodeEnded)
        }
        _ = try env.reset()
        _ = try env.step(0)
        try env.close()
        XCTAssertThrowsError(try env.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
        XCTAssertThrowsError(try env.step(0)) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }

    func testAnyEnvironmentForwardsAndErases() throws {
        let env = CounterEnv()
        let any = AnyEnvironment(env)
        XCTAssertEqual(any.spec?.id, "Counter-v0")
        XCTAssertTrue(any.observationSpace.contains(0))
        XCTAssertTrue(any.actionSpace.contains(2))
        XCTAssertFalse(any.actionSpace.contains(9))

        let reset = try any.reset(seed: Seed(7))
        XCTAssertEqual(reset.observation as? Int, 0)

        let step = try any.step(2)
        XCTAssertEqual(step.observation as? Int, 2)
        XCTAssertEqual(step.reward, 2)

        XCTAssertThrowsError(try any.step("bad")) { err in
            guard case .invalidAction = err as? EnvironmentError else {
                return XCTFail("expected invalidAction, got \(err)")
            }
        }

        XCTAssertTrue(any.unwrapped === any)
        try any.close()
        XCTAssertThrowsError(try any.reset()) { err in
            XCTAssertEqual(err as? EnvironmentError, .closed)
        }
    }

    func testAnyEnvironmentInvalidActionCase() throws {
        let any = AnyEnvironment(CounterEnv())
        _ = try any.reset()
        XCTAssertThrowsError(try any.step(1.5 as Double)) { err in
            guard case .invalidAction = err as? EnvironmentError else {
                return XCTFail("expected invalidAction, got \(err)")
            }
        }
    }

    func testResetConvenienceOnProtocol() throws {
        let env = CounterEnv()
        let r = try env.reset()
        XCTAssertEqual(r.observation, 0)
        let r2 = try env.reset(seed: Seed(1))
        XCTAssertEqual(r2.observation, 0)
        XCTAssertEqual(env.debugLastSeed, 1)
    }
}
