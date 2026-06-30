// VectorEnvironmentError — failures for vector env APIs (design.md §20.1, PR-14).

/// Errors thrown by vector environment APIs (`SyncVectorEnv`, ``AsyncVectorEnv``).
public enum VectorEnvironmentError: Error, Equatable, Sendable {
    /// Operation on a vector env that has been `close()`d.
    case closed
    /// Outstanding reset/step work was cancelled (e.g. by ``AsyncVectorEnv/close()``).
    case cancelled
    /// Batched action/observation count did not match ``numEnvs``.
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
