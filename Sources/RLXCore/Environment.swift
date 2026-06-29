// Environment — synchronous MDP interaction protocol (design.md §8.2, §11, Appendix A).
//
// Class-bound: envs hold mutable episode state. Construction does not start an episode;
// first legal call is `reset`. After terminated || truncated, caller must `reset` before
// `step` (enforced by stubs/checkers; core protocol documents only). `close` is idempotent;
// further reset/step should throw `EnvironmentError.closed`.

/// Optional knobs for ``Environment/reset(seed:options:)`` (curriculum, fixed init, …).
///
/// Concrete envs may define their own `ResetOptions` conformers; core stays agnostic.
public protocol ResetOptions: Sendable {}

/// Synchronous single-environment interaction surface (PR-06).
///
/// - Observation/action types are tied to ``observationSpace`` / ``actionSpace`` `Value`.
/// - Reward on ``StepResult`` is always ``Float`` (MLX float32 alignment).
/// - Prefer seeding via `reset(seed:)` with an explicit `UInt64?` (use ``Seed/rawValue``).
/// - Do not call process-global `MLXRandom.seed` inside env implementations.
public protocol Environment: AnyObject {
    associatedtype Observation
    associatedtype Action
    associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation
    associatedtype ActionSpace: Space where ActionSpace.Value == Action

    var observationSpace: ObservationSpace { get }
    var actionSpace: ActionSpace { get }

    /// Kind metadata; `nil` if the instance is ad hoc / unregistered.
    var spec: EnvSpec? { get }

    /// Start or restart an episode.
    ///
    /// - Parameters:
    ///   - seed: If non-`nil`, re-seed env PRNG streams derived from this value; if `nil`,
    ///     continue the previous RNG sequence (design.md §11.1).
    ///   - options: Env-specific reset knobs.
    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Observation>

    /// Apply `action` and return the transition outcome.
    func step(_ action: Action) throws -> StepResult<Observation>

    /// Release resources. Safe to call more than once; subsequent interaction throws ``EnvironmentError/closed``.
    func close() throws
}

extension Environment {
    /// Default: no kind metadata.
    public var spec: EnvSpec? { nil }

    /// Convenience: `reset` with no seed or options.
    public func reset() throws -> ResetResult<Observation> {
        try reset(seed: nil, options: nil)
    }

    /// Convenience: `reset` with a ``Seed``.
    public func reset(seed: Seed, options: (any ResetOptions)? = nil) throws -> ResetResult<Observation> {
        try reset(seed: seed.rawValue, options: options)
    }
}

/// Optional rendering capability (design.md §17). Not required on every env.
public protocol Renderable: AnyObject {
    func render() throws -> RenderFrame?
}
