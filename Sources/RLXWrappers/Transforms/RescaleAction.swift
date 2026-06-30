// RescaleAction — linear map policy Box → env Box (design.md §15.2, PR-09).

import MLX
import RLXCore

/// Exposes a **policy** ``BoxSpace`` as ``actionSpace`` and maps actions linearly into the
/// inner env's box before ``step``.
///
/// Mapping (elementwise):
/// `env = low + (a - min) * (high - low) / (max - min)`
/// with `min`/`max` from the policy space and `low`/`high` from the env space.
///
/// Requires finite, matching shapes and ``Inner/Action`` == ``MLXArray``.
public final class RescaleAction<Inner: Environment>: Environment, EnvironmentWrapper
where Inner.Action == MLXArray, Inner.ActionSpace == BoxSpace {
    public typealias Observation = Inner.Observation
    public typealias Action = MLXArray
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = BoxSpace

    public let inner: Inner
    /// Policy-facing action space (what the agent outputs).
    public let actionSpace: BoxSpace

    public init(_ inner: Inner, policyActionSpace: BoxSpace) {
        precondition(
            policyActionSpace.shape == inner.actionSpace.shape,
            "policy and env action spaces must have the same shape"
        )
        self.inner = inner
        self.actionSpace = policyActionSpace
    }

    /// Convenience: policy actions in `[min, max]` broadcast to the env action shape.
    public convenience init(_ inner: Inner, min: Float, max: Float) {
        precondition(min < max, "policy min must be < max")
        let shape = inner.actionSpace.shape ?? []
        let dtype = inner.actionSpace.dtype ?? .float32
        let policy = BoxSpace(low: min, high: max, shape: shape, dtype: dtype)
        self.init(inner, policyActionSpace: policy)
    }

    public var observationSpace: ObservationSpace { inner.observationSpace }
    public var spec: EnvSpec? { inner.spec }

    public var unwrapped: AnyEnvironment { AnyEnvironment(inner) }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Observation> {
        try inner.reset(seed: seed, options: options)
    }

    public func step(_ action: MLXArray) throws -> StepResult<Observation> {
        let envSpace = inner.actionSpace
        let policyMin = actionSpace.low
        let policyMax = actionSpace.high
        let envLow = envSpace.low
        let envHigh = envSpace.high
        let span = policyMax - policyMin
        // Avoid div-by-zero on degenerate axes (should not happen if min < max scalars).
        let scaled = envLow + (action - policyMin) * (envHigh - envLow) / span
        return try inner.step(scaled)
    }

    public func close() throws {
        try inner.close()
    }
}
