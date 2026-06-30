// SyncVectorEnv — sequential vector of AnyEnvironment (design.md §16.2–16.4, PR-13).

import RLXCore
import RLXWrappers

/// Synchronously steps `numEnvs` independent ``AnyEnvironment`` instances in fixed index order.
///
/// - **Autoreset:** see ``AutoresetMode`` (default ``AutoresetMode/sameStep``).
/// - **Seeding:** `reset(seed:)` uses `Seed(seed).child(index: i)` per slot when `seed != nil`.
/// - Not thread-safe; call from one task at a time.
public final class SyncVectorEnv {
    public let numEnvs: Int
    public let autoresetMode: AutoresetMode

    private var envs: [AnyEnvironment]
    /// For ``AutoresetMode/nextStep``: slot must reset before the next step action.
    private var pendingReset: [Bool]

    /// - Parameters:
    ///   - numEnvs: Number of parallel logical environments (must be > 0).
    ///   - autoresetMode: Episode boundary policy (immutable).
    ///   - makeEnv: Factory invoked `numEnvs` times (fresh instance each call).
    public init(
        numEnvs: Int,
        autoresetMode: AutoresetMode = .sameStep,
        makeEnv: () -> AnyEnvironment
    ) {
        precondition(numEnvs > 0, "numEnvs must be > 0")
        self.numEnvs = numEnvs
        self.autoresetMode = autoresetMode
        self.envs = (0..<numEnvs).map { _ in makeEnv() }
        self.pendingReset = Array(repeating: false, count: numEnvs)
    }

    /// Convenience: `numEnvs` copies from a factory that takes a slot index (for varied config).
    public init(
        numEnvs: Int,
        autoresetMode: AutoresetMode = .sameStep,
        makeEnv: (Int) -> AnyEnvironment
    ) {
        precondition(numEnvs > 0, "numEnvs must be > 0")
        self.numEnvs = numEnvs
        self.autoresetMode = autoresetMode
        self.envs = (0..<numEnvs).map { makeEnv($0) }
        self.pendingReset = Array(repeating: false, count: numEnvs)
    }

    public var singleObservationSpace: AnySpace { envs[0].observationSpace }
    public var singleActionSpace: AnySpace { envs[0].actionSpace }

    /// Reset all sub-environments.
    ///
    /// When `seed` is non-`nil`, slot `i` receives `Seed(seed).child(index: i).rawValue`.
    public func reset(
        seed: UInt64? = nil,
        options: (any ResetOptions)? = nil
    ) throws -> VectorResetResult {
        var observations: [Any] = []
        var infos: [Info] = []
        observations.reserveCapacity(numEnvs)
        infos.reserveCapacity(numEnvs)
        for i in 0..<numEnvs {
            let slotSeed: UInt64?
            if let seed {
                slotSeed = Seed(seed).child(index: i).rawValue
            } else {
                slotSeed = nil
            }
            let r = try envs[i].reset(seed: slotSeed, options: options)
            observations.append(r.observation)
            infos.append(r.info)
            pendingReset[i] = false
        }
        return VectorResetResult(observations: observations, infos: infos)
    }

    /// Step all environments with one action per slot (same length as ``numEnvs``).
    public func step(_ actions: [Any]) throws -> VectorStepResult {
        precondition(actions.count == numEnvs, "actions.count must equal numEnvs")
        var observations: [Any] = []
        var rewards: [Float] = []
        var terminateds: [Bool] = []
        var truncateds: [Bool] = []
        var infos: [Info] = []
        observations.reserveCapacity(numEnvs)
        rewards.reserveCapacity(numEnvs)
        terminateds.reserveCapacity(numEnvs)
        truncateds.reserveCapacity(numEnvs)
        infos.reserveCapacity(numEnvs)

        for i in 0..<numEnvs {
            if autoresetMode == .nextStep, pendingReset[i] {
                _ = try envs[i].reset(seed: nil, options: nil)
                pendingReset[i] = false
            }

            let step = try envs[i].step(actions[i])
            var obs = step.observation
            var info = step.info
            let terminated = step.terminated
            let truncated = step.truncated
            let done = terminated || truncated

            if done {
                switch autoresetMode {
                case .disabled:
                    break
                case .nextStep:
                    pendingReset[i] = true
                case .sameStep:
                    FinalObservationInfo.embed(step.observation, into: &info)
                    info[InfoKeys.finalInfo] = .nested(step.info)
                    let r = try envs[i].reset(seed: nil, options: nil)
                    obs = r.observation
                    // Keep terminal flags on this transition; live obs is next episode.
                }
            }

            observations.append(obs)
            rewards.append(step.reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        }

        return VectorStepResult(
            observations: observations,
            rewards: rewards,
            terminateds: terminateds,
            truncateds: truncateds,
            infos: infos
        )
    }

    public func close() throws {
        for env in envs {
            try env.close()
        }
    }
}
