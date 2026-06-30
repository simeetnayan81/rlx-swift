// EnvironmentRegistry — register / make / list (design.md §8.7, §18, PR-10).

import Foundation

/// Marker for env-specific construction knobs passed through ``EnvironmentRegistry/make``.
///
/// Concrete configs (e.g. future `CartPoleConfig`) conform and are type-checked inside factories.
public protocol EnvConfig: Sendable {}

/// Builds a type-erased environment from optional config and render mode.
public protocol EnvironmentFactory: Sendable {
    func make(config: (any EnvConfig)?, renderMode: RenderMode) throws -> AnyEnvironment
}

/// Closure-backed factory for tests and simple registrations.
public struct ClosureEnvironmentFactory: EnvironmentFactory {
    private let body: @Sendable ((any EnvConfig)?, RenderMode) throws -> AnyEnvironment

    public init(
        _ body: @escaping @Sendable ((any EnvConfig)?, RenderMode) throws -> AnyEnvironment
    ) {
        self.body = body
    }

    public func make(config: (any EnvConfig)?, renderMode: RenderMode) throws -> AnyEnvironment {
        try body(config, renderMode)
    }
}

/// Errors from registration and ``EnvironmentRegistry/make``.
public enum RegistryError: Error, Equatable, Sendable {
    /// No factory registered under this id.
    case unknownID(String)
    /// `register` called again with the same id.
    case duplicateID(String)
    /// Factory rejected the provided config (wrong type or invalid values).
    case invalidConfig(String)
    /// Other registration / construction failure.
    case configuration(String)
}

extension RegistryError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unknownID(let id):
            return "RegistryError.unknownID: \(id)"
        case .duplicateID(let id):
            return "RegistryError.duplicateID: \(id)"
        case .invalidConfig(let detail):
            return "RegistryError.invalidConfig: \(detail)"
        case .configuration(let detail):
            return "RegistryError.configuration: \(detail)"
        }
    }
}

/// Catalog of environment kinds: id → (``EnvSpec``, factory).
///
/// Registration is serialized. Prefer completing all `register` calls before multi-threaded
/// `make` (design.md §21). Concurrent `make` after registration is supported via a lock.
public final class EnvironmentRegistry: @unchecked Sendable {
    /// Process-wide default registry.
    public static let shared = EnvironmentRegistry()

    private struct Entry {
        var spec: EnvSpec
        var factory: any EnvironmentFactory
    }

    private var entries: [String: Entry] = [:]
    private let lock = NSLock()

    public init() {}

    /// Register a factory under a stable id. Throws ``RegistryError/duplicateID`` if already present.
    public func register(id: String, spec: EnvSpec, factory: some EnvironmentFactory) throws {
        lock.lock()
        defer { lock.unlock() }
        if entries[id] != nil {
            throw RegistryError.duplicateID(id)
        }
        // Prefer catalog id matching register key when callers pass consistent specs.
        var storedSpec = spec
        if storedSpec.id != id {
            storedSpec.id = id
        }
        entries[id] = Entry(spec: storedSpec, factory: factory)
    }

    /// Construct an environment for `id`.
    ///
    /// - Parameters:
    ///   - id: Registered kind id (e.g. `"DummyEnv-v0"`).
    ///   - config: Optional env-specific config; factories validate type.
    ///   - renderMode: Preferred render mode (factories may ignore if unsupported).
    public func make(
        _ id: String,
        config: (any EnvConfig)? = nil,
        renderMode: RenderMode = .none
    ) throws -> AnyEnvironment {
        let factory: any EnvironmentFactory
        lock.lock()
        guard let entry = entries[id] else {
            lock.unlock()
            throw RegistryError.unknownID(id)
        }
        factory = entry.factory
        lock.unlock()
        return try factory.make(config: config, renderMode: renderMode)
    }

    /// Sorted registered ids.
    public var ids: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries.keys.sorted()
    }

    /// Spec for a registered id, if any.
    public func spec(for id: String) -> EnvSpec? {
        lock.lock()
        defer { lock.unlock() }
        return entries[id]?.spec
    }
}
