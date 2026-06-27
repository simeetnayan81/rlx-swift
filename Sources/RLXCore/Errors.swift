// EnvironmentError — failure modes for single-env interaction (design.md §20.1).

/// Errors thrown by single-environment APIs (`reset`, `step`, `close`, `render`).
///
/// Vector, space, and registry errors are separate types introduced with those
/// subsystems; this enum covers the core interaction surface only.
public enum EnvironmentError: Error, Equatable, Sendable {
    /// `step` (or similar) called before the first successful `reset`.
    case notReset
    /// `step` called after a terminal transition without an intervening `reset`.
    case episodeEnded
    /// Operation on an environment that has been `close()`d.
    case closed
    /// Action outside the action space or otherwise rejected by dynamics.
    case invalidAction(String)
    /// Observation failed validation (e.g. passive checker / space contains).
    case invalidObservation(String)
    /// `render()` requested but the env was constructed with `.none` or lacks support.
    case renderNotSupported
    /// Invalid construction or runtime configuration.
    case configuration(String)
    /// Wrapped failure from an underlying system (I/O, physics, …).
    ///
    /// Equality compares only the case identity (both `.underlying`), not the
    /// nested `Error` values — `Error` is not `Equatable`.
    case underlying(any Error)

    public static func == (lhs: EnvironmentError, rhs: EnvironmentError) -> Bool {
        switch (lhs, rhs) {
        case (.notReset, .notReset),
             (.episodeEnded, .episodeEnded),
             (.closed, .closed),
             (.renderNotSupported, .renderNotSupported):
            return true
        case (.invalidAction(let a), .invalidAction(let b)),
             (.invalidObservation(let a), .invalidObservation(let b)),
             (.configuration(let a), .configuration(let b)):
            return a == b
        case (.underlying, .underlying):
            // Nested errors are not Equatable; treat same case as equal.
            return true
        default:
            return false
        }
    }
}

extension EnvironmentError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .notReset:
            return "EnvironmentError.notReset: call reset() before step()"
        case .episodeEnded:
            return "EnvironmentError.episodeEnded: call reset() after a terminal transition"
        case .closed:
            return "EnvironmentError.closed: environment has been closed"
        case .invalidAction(let detail):
            return "EnvironmentError.invalidAction: \(detail)"
        case .invalidObservation(let detail):
            return "EnvironmentError.invalidObservation: \(detail)"
        case .renderNotSupported:
            return "EnvironmentError.renderNotSupported"
        case .configuration(let detail):
            return "EnvironmentError.configuration: \(detail)"
        case .underlying(let error):
            return "EnvironmentError.underlying: \(error)"
        }
    }
}
