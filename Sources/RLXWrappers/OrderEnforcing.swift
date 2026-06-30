// OrderEnforcing — opt-in lifecycle enforcement (design.md §8.2, §15.2, §20.2).

import RLXCore

/// Throws ``EnvironmentError/notReset`` or ``EnvironmentError/episodeEnded`` when
/// callers violate reset/step ordering, regardless of whether `inner` enforces it.
///
/// Tracks episode boundaries using ``StepResult/done`` (`terminated || truncated`).
public final class OrderEnforcing<Inner: Environment>: Environment, EnvironmentWrapper {
    public typealias Observation = Inner.Observation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner

    private var hasReset = false
    /// `true` after a step with `terminated || truncated` until the next successful reset.
    private var needsReset = false

    public init(_ inner: Inner) {
        self.inner = inner
    }

    public var observationSpace: ObservationSpace { inner.observationSpace }
    public var actionSpace: ActionSpace { inner.actionSpace }
    public var spec: EnvSpec? { inner.spec }

    /// Single-layer unwrap to the immediate inner env (multi-layer stacks refine later).
    public var unwrapped: AnyEnvironment {
        AnyEnvironment(inner)
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Observation> {
        let result = try inner.reset(seed: seed, options: options)
        hasReset = true
        needsReset = false
        return result
    }

    public func step(_ action: Action) throws -> StepResult<Observation> {
        guard hasReset else { throw EnvironmentError.notReset }
        guard !needsReset else { throw EnvironmentError.episodeEnded }
        let result = try inner.step(action)
        if result.done {
            needsReset = true
        }
        return result
    }

    public func close() throws {
        try inner.close()
    }
}
