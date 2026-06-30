// ClipAction — clamp Box actions to bounds before inner step (design.md §15.2, PR-09).

import MLX
import RLXCore

/// Clips each action element to the inner ``BoxSpace`` low/high before forwarding ``step``.
///
/// Observation path is unchanged. Requires ``Inner/Action`` == ``MLXArray`` and
/// ``Inner/ActionSpace`` == ``BoxSpace``.
public final class ClipAction<Inner: Environment>: Environment, EnvironmentWrapper
where Inner.Action == MLXArray, Inner.ActionSpace == BoxSpace {
    public typealias Observation = Inner.Observation
    public typealias Action = MLXArray
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = BoxSpace

    public let inner: Inner

    public init(_ inner: Inner) {
        self.inner = inner
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

    public func step(_ action: MLXArray) throws -> StepResult<Observation> {
        let space = inner.actionSpace
        let clipped = clip(action, space.low, space.high)
        return try inner.step(clipped)
    }

    public func close() throws {
        try inner.close()
    }
}

/// Elementwise clamp `x` into `[low, high]` (MLX `maximum` / `minimum`).
private func clip(_ x: MLXArray, _ low: MLXArray, _ high: MLXArray) -> MLXArray {
    minimum(maximum(x, low), high)
}
