// EnvironmentWrapper — composable env adapter protocol (design.md §8.3, §15.1, Appendix A).

import RLXCore

/// Marker / structural protocol for environments that wrap an ``inner`` env.
///
/// Conformers forward spaces, `spec`, `reset`, `step`, and `close` unless they
/// intentionally override behaviour (e.g. ``OrderEnforcing``, TimeLimit in PR-08).
///
/// Associated observation/action types typically match `Inner` when the wrapper
/// does not transform them. Transforming wrappers (PR-09) may use different types
/// and need not use the default type-equality pattern of order/lifecycle wrappers.
public protocol EnvironmentWrapper: Environment {
    associatedtype Inner: Environment
    /// Immediate wrapped environment (one layer down).
    var inner: Inner { get }
}
