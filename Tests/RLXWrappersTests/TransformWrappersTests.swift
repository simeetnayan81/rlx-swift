import MLX
import RLXCore
import RLXEnvs
import RLXWrappers
import XCTest

// MARK: - Minimal Box action env for Clip / Rescale tests

/// Identity-ish env: obs = action (clipped by wrappers), reward = sum of action elements.
private final class BoxIdentityEnv: Environment {
    typealias Observation = MLXArray
    typealias Action = MLXArray
    typealias ObservationSpace = BoxSpace
    typealias ActionSpace = BoxSpace

    let observationSpace: BoxSpace
    let actionSpace: BoxSpace

    private var hasReset = false
    private var steps = 0
    private let episodeLength: Int

    init(low: Float = -2, high: Float = 2, shape: [Int] = [2], episodeLength: Int = 10) {
        self.observationSpace = BoxSpace(low: low, high: high, shape: shape)
        self.actionSpace = BoxSpace(low: low, high: high, shape: shape)
        self.episodeLength = episodeLength
    }

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<MLXArray> {
        _ = seed
        _ = options
        hasReset = true
        steps = 0
        let z = MLXArray.zeros(actionSpace.shape ?? [2], dtype: .float32)
        return ResetResult(observation: z)
    }

    func step(_ action: MLXArray) throws -> StepResult<MLXArray> {
        if !hasReset { throw EnvironmentError.notReset }
        steps += 1
        let reward: Float = Device.withDefaultDevice(.cpu) {
            eval(action)
            return action.asArray(Float.self).reduce(0, +)
        }
        let terminated = steps >= episodeLength
        return StepResult(
            observation: action,
            reward: reward,
            terminated: terminated,
            truncated: false
        )
    }

    func close() throws {}
}

private func floats(_ a: MLXArray) -> [Float] {
    Device.withDefaultDevice(.cpu) {
        eval(a)
        return a.asArray(Float.self)
    }
}

// MARK: - ClipAction

final class ClipActionTests: XCTestCase {

    func testClipsOutOfBoundsActions() throws {
        let env = ClipAction(BoxIdentityEnv(low: -1, high: 1, shape: [2], episodeLength: 3))
        _ = try env.reset()
        let out = MLXArray([Float(-5), Float(3)])
        let step = try env.step(out)
        let obs = floats(step.observation)
        XCTAssertEqual(obs[0], -1, accuracy: 1e-5)
        XCTAssertEqual(obs[1], 1, accuracy: 1e-5)
        // reward uses clipped action
        XCTAssertEqual(step.reward, 0, accuracy: 1e-5)
    }

    func testInBoundsUnchanged() throws {
        let env = ClipAction(BoxIdentityEnv(low: -2, high: 2, shape: [2]))
        _ = try env.reset()
        let a = MLXArray([Float(0.5), Float(-0.25)])
        let step = try env.step(a)
        let obs = floats(step.observation)
        XCTAssertEqual(obs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(obs[1], -0.25, accuracy: 1e-5)
    }

    func testActionSpaceMatchesInner() {
        let inner = BoxIdentityEnv(low: -1, high: 1, shape: [3])
        let env = ClipAction(inner)
        XCTAssertEqual(env.actionSpace.shape, [3])
        XCTAssertEqual(floats(env.actionSpace.low), floats(inner.actionSpace.low))
    }
}

// MARK: - RescaleAction

final class RescaleActionTests: XCTestCase {

    func testMapsPolicyMinusOneOneToEnvBox() throws {
        // Policy [-1,1] → env [0, 10], scalar shape [1]
        let inner = BoxIdentityEnv(low: 0, high: 10, shape: [1], episodeLength: 5)
        let env = RescaleAction(inner, min: -1, max: 1)
        _ = try env.reset()
        // policy action -1 → 0
        let lo = try env.step(MLXArray([Float(-1)]))
        XCTAssertEqual(floats(lo.observation)[0], 0, accuracy: 1e-4)
        // policy action 1 → 10
        let hi = try env.step(MLXArray([Float(1)]))
        XCTAssertEqual(floats(hi.observation)[0], 10, accuracy: 1e-4)
        // policy action 0 → 5
        let mid = try env.step(MLXArray([Float(0)]))
        XCTAssertEqual(floats(mid.observation)[0], 5, accuracy: 1e-4)
    }

    func testPolicyActionSpaceExposed() {
        let inner = BoxIdentityEnv(low: 0, high: 1, shape: [2])
        let env = RescaleAction(inner, min: -1, max: 1)
        XCTAssertEqual(floats(env.actionSpace.low), [-1, -1])
        XCTAssertEqual(floats(env.actionSpace.high), [1, 1])
        XCTAssertEqual(floats(env.inner.actionSpace.low), [0, 0])
    }
}

// MARK: - TransformObservation / TransformReward

final class TransformObservationRewardTests: XCTestCase {

    func testTransformObservationMapsResetAndStep() throws {
        // Double DummyEnv Int obs; new space still Discrete(n: 10) so contains holds for small values
        let env = TransformObservation(
            DummyEnv(observationN: 5, actionN: 5, episodeLength: 4),
            observationSpace: DiscreteSpace(n: 10)
        ) { obs in obs * 2 }

        let r = try env.reset()
        XCTAssertEqual(r.observation, 0)
        let s = try env.step(2) // obs becomes 2, transformed 4
        XCTAssertEqual(s.observation, 4)
        XCTAssertTrue(env.observationSpace.contains(s.observation))
        XCTAssertEqual(s.reward, 2) // untransformed reward
    }

    func testTransformRewardScales() throws {
        let env = TransformReward(DummyEnv(episodeLength: 3)) { $0 * 10 }
        _ = try env.reset()
        let s = try env.step(2)
        XCTAssertEqual(s.reward, 20)
        XCTAssertEqual(s.observation, 2)
    }

    func testTransformRewardSign() throws {
        let env = TransformReward(DummyEnv(episodeLength: 2)) { -$0 }
        _ = try env.reset()
        let s = try env.step(3)
        XCTAssertEqual(s.reward, -3)
    }

    func testComposeClipRescale() throws {
        // Policy [-1,1] → env [-2,2], then clip (no-op if rescale stays in bounds)
        let inner = BoxIdentityEnv(low: -2, high: 2, shape: [1])
        let env = ClipAction(RescaleAction(inner, min: -1, max: 1))
        _ = try env.reset()
        let s = try env.step(MLXArray([Float(1)]))
        XCTAssertEqual(floats(s.observation)[0], 2, accuracy: 1e-4)
    }

    func testTransformObservationComposeWithReward() throws {
        let env = TransformReward(
            TransformObservation(
                DummyEnv(episodeLength: 2),
                observationSpace: DiscreteSpace(n: 20)
            ) { $0 + 1 }
        ) { $0 + 0.5 }

        _ = try env.reset()
        let s = try env.step(1)
        XCTAssertEqual(s.observation, 2) // 0+1 then +1 from action dynamics? reset 0 -> transform 1; step action 1 -> obs 1 -> transform 2
        XCTAssertEqual(s.reward, 1.5)
    }
}
