// CartPole-v1 — classic control pole balancing (design.md §24.1, PR-11).
//
// Dynamics follow the standard inverted-pendulum / cart-pole equations used by
// common RL benchmarks (Euler integration, τ = 0.02). Observation is MLXArray
// shape [4]: [x, x_dot, theta, theta_dot]. Action is Discrete(2): 0 = left, 1 = right.

import Foundation
import MLX
import RLXCore
import RLXWrappers

/// Optional construction knobs for ``CartPoleEnv``.
public struct CartPoleConfig: EnvConfig, Sendable, Equatable {
    public var gravity: Float
    public var massCart: Float
    public var massPole: Float
    public var length: Float
    public var forceMag: Float
    public var tau: Float
    /// Half-width of uniform reset noise on each state component.
    public var resetNoise: Float

    public init(
        gravity: Float = 9.8,
        massCart: Float = 1.0,
        massPole: Float = 0.1,
        length: Float = 0.5,
        forceMag: Float = 10.0,
        tau: Float = 0.02,
        resetNoise: Float = 0.05
    ) {
        self.gravity = gravity
        self.massCart = massCart
        self.massPole = massPole
        self.length = length
        self.forceMag = forceMag
        self.tau = tau
        self.resetNoise = resetNoise
    }

    public static let `default` = CartPoleConfig()
}

/// Cart-pole balancing environment (`CartPole-v1`).
///
/// Episode ends with ``StepResult/terminated`` when `|x| > 2.4` or
/// `|theta| > 12°` (≈ 0.2095 rad). Reward is `+1` on every step.
/// Time limits are applied by ``TimeLimit`` in the registered factory stack (max 500).
public final class CartPoleEnv: Environment {
    public typealias Observation = MLXArray
    public typealias Action = Int
    public typealias ObservationSpace = BoxSpace
    public typealias ActionSpace = DiscreteSpace

    public let observationSpace: BoxSpace
    public let actionSpace: DiscreteSpace
    public let config: CartPoleConfig
    public let renderMode: RenderMode
    public let spec: EnvSpec?

    /// Termination thresholds (task MDP).
    public static let xThreshold: Float = 2.4
    public static let thetaThresholdRadians: Float = 12 * .pi / 180

    private var state: (x: Float, xDot: Float, theta: Float, thetaDot: Float) = (0, 0, 0, 0)
    private var rng = SplitMix64(seed: 0)
    private var hasReset = false
    private var episodeOver = false
    private var isClosed = false

    public init(config: CartPoleConfig = .default, renderMode: RenderMode = .none) {
        self.config = config
        self.renderMode = renderMode
        // Observation bounds: generous box; episode terminates earlier at thresholds.
        let low = MLXArray([Float(-4.8), -Float.greatestFiniteMagnitude, -0.418, -Float.greatestFiniteMagnitude])
        let high = MLXArray([Float(4.8), Float.greatestFiniteMagnitude, 0.418, Float.greatestFiniteMagnitude])
        self.observationSpace = BoxSpace(low: low, high: high, dtype: .float32)
        self.actionSpace = DiscreteSpace(n: 2)
        self.spec = EnvSpec(
            id: "CartPole-v1",
            maxEpisodeSteps: 500,
            rewardThreshold: 475,
            nondeterministic: false,
            defaultRenderMode: renderMode,
            version: 1
        )
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<MLXArray> {
        if isClosed { throw EnvironmentError.closed }
        _ = options
        if let seed {
            rng = SplitMix64(seed: seed)
        }
        let n = config.resetNoise
        state = (
            uniform(-n, n),
            uniform(-n, n),
            uniform(-n, n),
            uniform(-n, n)
        )
        hasReset = true
        episodeOver = false
        return ResetResult(observation: observationArray())
    }

    public func step(_ action: Int) throws -> StepResult<MLXArray> {
        if isClosed { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        if episodeOver { throw EnvironmentError.episodeEnded }
        guard actionSpace.contains(action) else {
            throw EnvironmentError.invalidAction("CartPole action must be 0 or 1, got \(action)")
        }

        let force: Float = action == 1 ? config.forceMag : -config.forceMag
        let (x, xDot, theta, thetaDot) = state
        let g = config.gravity
        let mc = config.massCart
        let mp = config.massPole
        let totalMass = mc + mp
        let length = config.length
        let poleMassLength = mp * length
        let cosTheta = cos(theta)
        let sinTheta = sin(theta)

        let temp = (force + poleMassLength * thetaDot * thetaDot * sinTheta) / totalMass
        let thetaAcc =
            (g * sinTheta - cosTheta * temp)
            / (length * (4.0 / 3.0 - mp * cosTheta * cosTheta / totalMass))
        let xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass

        let tau = config.tau
        let newX = x + tau * xDot
        let newXDot = xDot + tau * xAcc
        let newTheta = theta + tau * thetaDot
        let newThetaDot = thetaDot + tau * thetaAcc
        state = (newX, newXDot, newTheta, newThetaDot)

        let terminated =
            abs(newX) > Self.xThreshold
            || abs(newTheta) > Self.thetaThresholdRadians
        if terminated { episodeOver = true }

        return StepResult(
            observation: observationArray(),
            reward: 1.0,
            terminated: terminated,
            truncated: false
        )
    }

    public func close() throws {
        isClosed = true
    }

    private func uniform(_ lo: Float, _ hi: Float) -> Float {
        // Map UInt64 to [0,1) then scale — portable, no MLX.
        let u = Double(rng.next() >> 11) * (1.0 / Double(1 << 53))
        return lo + (hi - lo) * Float(u)
    }

    private func observationArray() -> MLXArray {
        MLXArray([state.x, state.xDot, state.theta, state.thetaDot])
    }
}

// MARK: - Factory helpers

extension CartPoleEnv {
    /// Build ``AnyEnvironment`` with optional default wrapper stack (order + time limit).
    public static func makeAny(
        config: CartPoleConfig = .default,
        renderMode: RenderMode = .none,
        orderEnforcing: Bool = true,
        maxEpisodeSteps: Int? = 500
    ) -> AnyEnvironment {
        let base = CartPoleEnv(config: config, renderMode: renderMode)
        if orderEnforcing, let maxEpisodeSteps {
            return AnyEnvironment(
                OrderEnforcing(TimeLimit(base, maxEpisodeSteps: maxEpisodeSteps))
            )
        }
        if orderEnforcing {
            return AnyEnvironment(OrderEnforcing(base))
        }
        if let maxEpisodeSteps {
            return AnyEnvironment(TimeLimit(base, maxEpisodeSteps: maxEpisodeSteps))
        }
        return AnyEnvironment(base)
    }
}
