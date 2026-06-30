// Pendulum-v1 — continuous torque control (design.md §24.1, PR-12).
//
// Dynamics aligned with Gymnasium classic_control/pendulum.py (default g=10, dt=0.05).
// Observation MLXArray [3]: [cos(theta), sin(theta), theta_dot].
// Action MLXArray [1]: torque in [-maxTorque, maxTorque] (default ±2).

import Foundation
import MLX
import RLXCore
import RLXWrappers

/// Construction knobs for ``PendulumEnv``.
public struct PendulumConfig: EnvConfig, Sendable, Equatable {
    public var gravity: Float
    public var mass: Float
    public var length: Float
    public var dt: Float
    public var maxSpeed: Float
    public var maxTorque: Float

    public init(
        gravity: Float = 10.0,
        mass: Float = 1.0,
        length: Float = 1.0,
        dt: Float = 0.05,
        maxSpeed: Float = 8.0,
        maxTorque: Float = 2.0
    ) {
        self.gravity = gravity
        self.mass = mass
        self.length = length
        self.dt = dt
        self.maxSpeed = maxSpeed
        self.maxTorque = maxTorque
    }

    public static let `default` = PendulumConfig()
}

/// Inverted pendulum with continuous torque (`Pendulum-v1`).
///
/// Does **not** set ``StepResult/terminated`` from dynamics (infinite-horizon task MDP).
/// Episode length uses ``TimeLimit`` in the registered factory (default 200 steps).
/// Reward: `-(angle_normalize(theta)^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)`.
public final class PendulumEnv: Environment {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    public typealias ObservationSpace = BoxSpace
    public typealias ActionSpace = BoxSpace

    public let observationSpace: BoxSpace
    public let actionSpace: BoxSpace
    public let config: PendulumConfig
    public let renderMode: RenderMode
    public let spec: EnvSpec?

    private var theta: Float = 0
    private var thetaDot: Float = 0
    private var lastTorque: Float = 0
    private var rng = SplitMix64(seed: 0)
    private var hasReset = false
    private var isClosed = false

    public init(config: PendulumConfig = .default, renderMode: RenderMode = .none) {
        self.config = config
        self.renderMode = renderMode
        // Obs bounds match Gymnasium Pendulum (cos/sin in [-1,1], speed in ±maxSpeed).
        let maxS = config.maxSpeed
        let oLow = MLXArray([Float(-1), Float(-1), -maxS])
        let oHigh = MLXArray([Float(1), Float(1), maxS])
        self.observationSpace = BoxSpace(low: oLow, high: oHigh, dtype: .float32)
        let t = config.maxTorque
        self.actionSpace = BoxSpace(low: -t, high: t, shape: [1], dtype: .float32)
        self.spec = EnvSpec(
            id: "Pendulum-v1",
            maxEpisodeSteps: 200,
            nondeterministic: false,
            defaultRenderMode: RenderMode.none,
            version: 1
        )
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<MLXArray> {
        if isClosed { throw EnvironmentError.closed }
        _ = options
        if let seed { rng = SplitMix64(seed: seed) }
        // Gymnasium: theta ~ U(-pi, pi), theta_dot ~ U(-1, 1)
        theta = uniform(-Float.pi, Float.pi)
        thetaDot = uniform(-1, 1)
        lastTorque = 0
        hasReset = true
        return ResetResult(observation: observationArray())
    }

    public func step(_ action: MLXArray) throws -> StepResult<MLXArray> {
        if isClosed { throw EnvironmentError.closed }
        if !hasReset { throw EnvironmentError.notReset }
        guard action.shape == [1] else {
            throw EnvironmentError.invalidAction("Pendulum action must have shape [1]")
        }
        // Clip torque to action bounds (Gymnasium clips u).
        let uRaw: Float = Device.withDefaultDevice(.cpu) {
            eval(action)
            return action.asArray(Float.self)[0]
        }
        let u = min(max(uRaw, -config.maxTorque), config.maxTorque)
        lastTorque = u

        let g = config.gravity
        let m = config.mass
        let l = config.length
        let dt = config.dt

        // Gymnasium pendulum dynamics
        var newThetaDot =
            thetaDot
            + (3 * g / (2 * l) * sin(theta) + 3 / (m * l * l) * u) * dt
        newThetaDot = min(max(newThetaDot, -config.maxSpeed), config.maxSpeed)
        let newTheta = theta + newThetaDot * dt

        theta = newTheta
        thetaDot = newThetaDot

        let ang = Self.angleNormalize(theta)
        let costs =
            ang * ang
            + 0.1 * thetaDot * thetaDot
            + 0.001 * u * u
        let reward = -costs

        return StepResult(
            observation: observationArray(),
            reward: reward,
            terminated: false,
            truncated: false
        )
    }

    public func close() throws {
        isClosed = true
    }

    /// Last applied torque (for tests / diagnostics).
    public var debugLastTorque: Float { lastTorque }
    public var debugTheta: Float { theta }
    public var debugThetaDot: Float { thetaDot }

    private func observationArray() -> MLXArray {
        MLXArray([cos(theta), sin(theta), thetaDot])
    }

    private func uniform(_ lo: Float, _ hi: Float) -> Float {
        let u = Double(rng.next() >> 11) * (1.0 / Double(1 << 53))
        return lo + (hi - lo) * Float(u)
    }

    /// Map angle to [-π, π] (Gymnasium `angle_normalize`: `((x + π) % (2π)) - π`).
    public static func angleNormalize(_ x: Float) -> Float {
        let twoPi = 2 * Float.pi
        var a = (x + Float.pi).truncatingRemainder(dividingBy: twoPi)
        if a < 0 { a += twoPi }
        return a - Float.pi
    }
}

extension PendulumEnv {
    /// Registered stack: order enforcing + time limit (200).
    public static func makeAny(
        config: PendulumConfig = .default,
        renderMode: RenderMode = .none,
        orderEnforcing: Bool = true,
        maxEpisodeSteps: Int? = 200
    ) -> AnyEnvironment {
        let base = PendulumEnv(config: config, renderMode: renderMode)
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
