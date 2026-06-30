// TransformReward — map scalar rewards (design.md §15.2, PR-09).

import RLXCore

/// Applies a pure function to ``StepResult/reward`` on each ``step``.
/// Observation and action paths are unchanged.
public final class TransformReward<Inner: Environment>: Environment, EnvironmentWrapper {
    public typealias Observation = Inner.Observation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner
    private let transform: (Float) -> Float

    public init(_ inner: Inner, transform: @escaping (Float) -> Float) {
        self.inner = inner
        self.transform = transform
    }

    public var observationSpace: ObservationSpace { inner.observationSpace }
    public var actionSpace: ActionSpace { inner.actionSpace }
    public var spec: EnvSpec? { inner.spec }

    public var unwrapped: AnyEnvironment { AnyEnvironment(inner) }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Observation> {
        try inner.reset(seed: seed, options: options)
    }

    public func step(_ action: Action) throws -> StepResult<Observation> {
        var r = try inner.step(action)
        r.reward = transform(r.reward)
        return r
    }

    public func close() throws {
        try inner.close()
    }
}
