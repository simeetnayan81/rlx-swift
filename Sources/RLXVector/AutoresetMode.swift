// AutoresetMode — vector episode boundary policy (design.md §16.4, PR-13).

/// Episode-boundary policy for vector environments (``SyncVectorEnv``, ``AsyncVectorEnv``).
///
/// Chosen at construction and **immutable** for the life of the vector env. Default in v1
/// is ``sameStep`` (convenient for batched training: live observation is always “current”
/// episode; terminal transition recoverable from info).
///
/// | Mode | On `terminated \|\| truncated` |
/// |------|--------------------------------|
/// | ``disabled`` | No auto reset; further `step` on that slot may throw |
/// | ``nextStep`` | Return terminal transition as-is; reset at the **start** of the next `step` for that slot |
/// | ``sameStep`` | Reset **inside** the ending step; return **new** episode’s first obs; stash terminal under info keys |
///
/// Design reference: `design.md` §16.5.
public enum AutoresetMode: String, Sendable, Equatable, CaseIterable {
    /// No automatic reset; caller must not step ended slots (env may throw ``EnvironmentError/episodeEnded``).
    case disabled
    /// Return the terminal transition as-is; reset on the **next** `step` for that slot.
    case nextStep
    /// Within the ending step, reset and return the **new** episode’s first observation;
    /// stash terminal obs/info under ``InfoKeys/finalObservation`` / ``InfoKeys/finalInfo``.
    case sameStep
}
