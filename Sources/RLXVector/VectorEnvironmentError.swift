// VectorEnvironmentError — failures for vector env APIs (design.md §20.1, PR-14).

/// Errors specific to **vector** environment APIs (``SyncVectorEnv``, ``AsyncVectorEnv``).
///
/// Single-env failures from a slot (e.g. ``EnvironmentError/invalidAction(_:)``) propagate as
/// the underlying error and fail the whole batch (v1: no per-lane `Result`).
public enum VectorEnvironmentError: Error, Equatable, Sendable {
    /// Operation after ``AsyncVectorEnv/close()`` (or equivalent) has completed.
    case closed
    /// Outstanding reset/step work was cancelled (typically ``AsyncVectorEnv/close()`` during an in-flight batch).
    case cancelled
    /// `actions.count` (or similar batch length) did not equal the vector’s `numEnvs`.
    case batchSizeMismatch(expected: Int, actual: Int)

    public static func == (lhs: VectorEnvironmentError, rhs: VectorEnvironmentError) -> Bool {
        switch (lhs, rhs) {
        case (.closed, .closed), (.cancelled, .cancelled):
            return true
        case (.batchSizeMismatch(let e1, let a1), .batchSizeMismatch(let e2, let a2)):
            return e1 == e2 && a1 == a2
        default:
            return false
        }
    }
}

extension VectorEnvironmentError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .closed:
            return "VectorEnvironmentError.closed: vector environment has been closed"
        case .cancelled:
            return "VectorEnvironmentError.cancelled: vector operation was cancelled"
        case .batchSizeMismatch(let expected, let actual):
            return "VectorEnvironmentError.batchSizeMismatch: expected \(expected), got \(actual)"
        }
    }
}
