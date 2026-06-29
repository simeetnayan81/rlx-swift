// AnyEnvironment — type eraser for heterogeneous envs / registry (design.md §27.4, PR-06).

/// Type-erased ``Environment`` for storage, `make`, and mixed collections.
///
/// Observation and action values are `Any`; spaces are ``AnySpace``. Wrong action types
/// on ``step`` yield ``EnvironmentError/invalidAction``.
///
/// ``unwrapped`` returns `self` for leaf envs; wrappers (PR-08) may override via a custom
/// initializer later. For PR-06, always `self`.
public final class AnyEnvironment: @unchecked Sendable {
    /// Type-erased observation space.
    public let observationSpace: AnySpace
    /// Type-erased action space.
    public let actionSpace: AnySpace
    /// Kind metadata from the concrete env, if any.
    public let spec: EnvSpec?

    private let _reset: (UInt64?, (any ResetOptions)?) throws -> ResetResult<Any>
    private let _step: (Any) throws -> StepResult<Any>
    private let _close: () throws -> Void

    /// Box a concrete environment.
    public init<E: Environment>(_ env: E) {
        self.observationSpace = AnySpace.erasing(env.observationSpace)
        self.actionSpace = AnySpace.erasing(env.actionSpace)
        self.spec = env.spec
        self._reset = { seed, options in
            let r = try env.reset(seed: seed, options: options)
            return ResetResult(observation: r.observation as Any, info: r.info)
        }
        self._step = { action in
            guard let typed = action as? E.Action else {
                throw EnvironmentError.invalidAction(
                    "expected \(String(describing: E.Action.self)), got \(String(describing: type(of: action)))"
                )
            }
            let r = try env.step(typed)
            return StepResult(
                observation: r.observation as Any,
                reward: r.reward,
                terminated: r.terminated,
                truncated: r.truncated,
                info: r.info
            )
        }
        self._close = {
            try env.close()
        }
    }

    /// Innermost env in a wrapper stack; leaf envs return `self` (PR-06).
    public var unwrapped: AnyEnvironment { self }

    /// Start or restart an episode (see ``Environment/reset(seed:options:)``).
    public func reset(
        seed: UInt64? = nil,
        options: (any ResetOptions)? = nil
    ) throws -> ResetResult<Any> {
        try _reset(seed, options)
    }

    /// Convenience with ``Seed``.
    public func reset(seed: Seed, options: (any ResetOptions)? = nil) throws -> ResetResult<Any> {
        try reset(seed: seed.rawValue, options: options)
    }

    /// Apply an action (`Any` must match the concrete action type).
    public func step(_ action: Any) throws -> StepResult<Any> {
        try _step(action)
    }

    /// Release resources (idempotent at the concrete env’s discretion; may throw if already closed).
    public func close() throws {
        try _close()
    }
}
