// AutoresetMode — vector episode boundary policy (design.md §16.4, PR-13).

/// How a vector env reacts when a sub-environment returns `terminated || truncated`.
public enum AutoresetMode: String, Sendable, Equatable, CaseIterable {
    /// No automatic reset; caller must not step ended slots (env may throw).
    case disabled
    /// Return the terminal transition as-is; reset on the **next** `step` for that slot.
    case nextStep
    /// Within the ending step, reset and return the **new** episode’s first observation;
    /// stash terminal obs/info under ``InfoKeys/finalObservation`` / ``InfoKeys/finalInfo``.
    case sameStep
}
