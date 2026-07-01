// checkEnvironment — multi-episode contract harness (design.md §20.3, §23.2).

import RLXCore
import RLXWrappers

/// Options for ``checkEnvironment(_:options:)``.
public struct CheckEnvironmentOptions: Sendable {
    /// Number of random-policy episodes to run.
    public var episodes: Int
    /// Safety cap on steps per episode (guards non-terminating envs).
    public var maxStepsPerEpisode: Int
    /// Seed for space sampling and env reset / determinism pair.
    public var seed: UInt64
    /// When `true`, wrap the env in ``OrderEnforcing`` for the post-terminal order check.
    public var enforceOrder: Bool

    public init(
        episodes: Int = 10,
        maxStepsPerEpisode: Int = 1000,
        seed: UInt64 = 0,
        enforceOrder: Bool = false
    ) {
        self.episodes = episodes
        self.maxStepsPerEpisode = maxStepsPerEpisode
        self.seed = seed
        self.enforceOrder = enforceOrder
    }
}

/// Failures raised by ``checkEnvironment(_:options:)`` (distinct from env interaction errors).
public enum CheckEnvironmentError: Error, Equatable, Sendable {
    case observationNotInSpace(String)
    case actionSampleNotInSpace
    case nonFiniteReward
    case expectedEpisodeEnded
    case determinismMismatch(String)
    case episodeTooLong(Int)
    case underlying(String)
}

/// Multi-episode **contract harness** for a concrete ``Environment`` (`design.md` §20.3).
///
/// Use in unit tests and CI. This is **not** a live wrapper — for per-step checks during
/// development, wrap with ``PassiveEnvChecker`` / ``OrderEnforcing`` instead (see
/// DocC *Validation layers*).
///
/// Exercises:
/// 1. Space sample + `contains`
/// 2. Reset / step produce in-space observations and finite rewards
/// 3. Optional post-terminal order check via ``OrderEnforcing``
/// 4. Determinism pair when the env is not marked nondeterministic
/// 5. Close idempotency
///
/// Requires `Observation: Equatable` so determinism can compare trajectories.
/// Prefer a **factory** so two independent instances can be constructed.
///
/// - Parameters:
///   - makeEnv: Fresh env factory (called multiple times).
///   - options: Episode counts, seed, whether to wrap with ``OrderEnforcing`` for order checks.
public func checkEnvironment<E: Environment>(
    _ makeEnv: () -> E,
    options: CheckEnvironmentOptions = CheckEnvironmentOptions()
) throws where E.Observation: Equatable {
    // 1. Spaces sample + contains
    var rng = SplitMix64(seed: options.seed)
    let probe = makeEnv()
    let obsSample = probe.observationSpace.sample(using: &rng)
    guard probe.observationSpace.contains(obsSample) else {
        throw CheckEnvironmentError.observationNotInSpace("sampled observation not in space")
    }
    let actSample = probe.actionSpace.sample(using: &rng)
    guard probe.actionSpace.contains(actSample) else {
        throw CheckEnvironmentError.actionSampleNotInSpace
    }
    try probe.close()

    // 2–3. Reset valid obs + random policy episodes
    let env = makeEnv()
    let reset0 = try env.reset(seed: options.seed, options: nil)
    guard env.observationSpace.contains(reset0.observation) else {
        throw CheckEnvironmentError.observationNotInSpace("reset observation not in space")
    }

    var actionsForDeterminism: [E.Action] = []
    var observationsForDeterminism: [E.Observation] = [reset0.observation]
    var rewardsForDeterminism: [Float] = []

    var episodeRng = SplitMix64(seed: options.seed &+ 1)
    for episode in 0..<options.episodes {
        if episode > 0 {
            let r = try env.reset(seed: nil, options: nil)
            guard env.observationSpace.contains(r.observation) else {
                throw CheckEnvironmentError.observationNotInSpace("reset observation not in space")
            }
        }
        var steps = 0
        var done = false
        while !done {
            if steps >= options.maxStepsPerEpisode {
                throw CheckEnvironmentError.episodeTooLong(steps)
            }
            let action = env.actionSpace.sample(using: &episodeRng)
            let step = try env.step(action)
            guard env.observationSpace.contains(step.observation) else {
                throw CheckEnvironmentError.observationNotInSpace("step observation not in space")
            }
            guard step.reward.isFinite else {
                throw CheckEnvironmentError.nonFiniteReward
            }
            if episode == 0 {
                actionsForDeterminism.append(action)
                observationsForDeterminism.append(step.observation)
                rewardsForDeterminism.append(step.reward)
            }
            done = step.done
            steps += 1
        }
    }

    // 4. After terminal, step without reset throws (self-enforcing env or OrderEnforcing)
    do {
        if options.enforceOrder {
            let ordered = OrderEnforcing(makeEnv())
            _ = try ordered.reset(seed: options.seed, options: nil)
            var orderRng = SplitMix64(seed: options.seed &+ 2)
            var done = false
            var guardSteps = 0
            while !done {
                if guardSteps >= options.maxStepsPerEpisode {
                    throw CheckEnvironmentError.episodeTooLong(guardSteps)
                }
                let action = ordered.actionSpace.sample(using: &orderRng)
                let step = try ordered.step(action)
                done = step.done
                guardSteps += 1
            }
            let extra = ordered.actionSpace.sample(using: &orderRng)
            do {
                _ = try ordered.step(extra)
                throw CheckEnvironmentError.expectedEpisodeEnded
            } catch let err as EnvironmentError where err == .episodeEnded {
                // expected
            } catch let err as CheckEnvironmentError {
                throw err
            } catch {
                throw CheckEnvironmentError.underlying(String(describing: error))
            }
            try ordered.close()
        } else {
            // env already finished last episode in loop above
            var extraRng = SplitMix64(seed: options.seed &+ 3)
            let extra = env.actionSpace.sample(using: &extraRng)
            do {
                _ = try env.step(extra)
                throw CheckEnvironmentError.expectedEpisodeEnded
            } catch let err as EnvironmentError where err == .episodeEnded {
                // expected
            } catch let err as CheckEnvironmentError {
                throw err
            } catch {
                throw CheckEnvironmentError.underlying(String(describing: error))
            }
        }
    }

    // 5. Determinism (skip if nondeterministic)
    let meta = makeEnv()
    let isNondeterministic = meta.spec?.nondeterministic == true
    try meta.close()
    if !isNondeterministic, !actionsForDeterminism.isEmpty {
        let a = makeEnv()
        let b = makeEnv()
        let ra = try a.reset(seed: options.seed, options: nil)
        let rb = try b.reset(seed: options.seed, options: nil)
        guard ra.observation == rb.observation else {
            throw CheckEnvironmentError.determinismMismatch("reset observations differ")
        }
        for action in actionsForDeterminism {
            let sa = try a.step(action)
            let sb = try b.step(action)
            guard sa.observation == sb.observation else {
                throw CheckEnvironmentError.determinismMismatch("step observations differ")
            }
            guard sa.reward == sb.reward else {
                throw CheckEnvironmentError.determinismMismatch("rewards differ")
            }
            guard sa.terminated == sb.terminated, sa.truncated == sb.truncated else {
                throw CheckEnvironmentError.determinismMismatch("done flags differ")
            }
            if sa.done { break }
        }
        try a.close()
        try b.close()
    }

    // 6. Close idempotency
    try env.close()
    try env.close()
    do {
        _ = try env.reset(seed: nil, options: nil)
        throw CheckEnvironmentError.underlying("expected closed after close(), reset succeeded")
    } catch let err as EnvironmentError where err == .closed {
        // expected
    } catch let err as CheckEnvironmentError {
        throw err
    } catch {
        throw CheckEnvironmentError.underlying(String(describing: error))
    }
}
