// EnvironmentWrapper — composable env adapter protocol (design.md §8.3, §15.1, Appendix A).

import RLXCore

/// Structural protocol for environments that adapt an ``inner`` env.
///
/// ## Requirements (design.md §15.1)
///
/// 1. Store ``inner``.
/// 2. Forward `observationSpace` / `actionSpace` / `spec` / `reset` / `step` / `close`
///    unless you intentionally change them.
/// 3. Override only what you change; call `inner` for the rest.
///
/// Lifecycle wrappers (``OrderEnforcing``, ``TimeLimit``, ``PassiveEnvChecker``) keep the
/// same observation/action types as `Inner`. Transform wrappers (``ClipAction``,
/// ``TransformObservation``, …) may change associated types and expose a new space.
///
/// Stack **outside-in** (outermost receives calls first). See DocC *Wrapper composition*
/// and repository `Documentation/DeveloperGuide.md`.
public protocol EnvironmentWrapper: Environment {
    associatedtype Inner: Environment
    /// Immediate wrapped environment (one layer down).
    var inner: Inner { get }
}
