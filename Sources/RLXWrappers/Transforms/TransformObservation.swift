// TransformObservation — map observations + new space (design.md §15.2, PR-09).

import RLXCore

/// Applies a pure transform to observations on ``reset`` and ``step``, exposing a new
/// ``observationSpace``. Action path is unchanged.
///
/// `NewObservation` may differ from ``Inner/Observation``. The transform should be
/// deterministic (design.md §15.4).
public final class TransformObservation<
    Inner: Environment,
    NewObservation,
    NewObservationSpace: Space
>: Environment, EnvironmentWrapper where NewObservationSpace.Value == NewObservation {
    public typealias Observation = NewObservation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = NewObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner
    public let observationSpace: NewObservationSpace
    private let transform: (Inner.Observation) throws -> NewObservation

    public init(
        _ inner: Inner,
        observationSpace: NewObservationSpace,
        transform: @escaping (Inner.Observation) throws -> NewObservation
    ) {
        self.inner = inner
        self.observationSpace = observationSpace
        self.transform = transform
    }

    public var actionSpace: ActionSpace { inner.actionSpace }
    public var spec: EnvSpec? { inner.spec }

    public var unwrapped: AnyEnvironment { AnyEnvironment(inner) }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<NewObservation> {
        let r = try inner.reset(seed: seed, options: options)
        return ResetResult(observation: try transform(r.observation), info: r.info)
    }

    public func step(_ action: Action) throws -> StepResult<NewObservation> {
        let r = try inner.step(action)
        return StepResult(
            observation: try transform(r.observation),
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            info: r.info
        )
    }

    public func close() throws {
        try inner.close()
    }
}
