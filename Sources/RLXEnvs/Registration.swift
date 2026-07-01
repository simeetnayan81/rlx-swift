// Registration — default factories into EnvironmentRegistry (design.md §18.1).

import RLXCore
import RLXWrappers

/// Registers built-in reference and debug environments on an ``EnvironmentRegistry``.
///
/// Call ``registerDefaults(on:)`` once at process start (or in tests). A second registration
/// of the same id throws ``RegistryError/duplicateID(_:)`` (default policy).
///
/// | Id | Implementation |
/// |----|-----------------|
/// | `DummyEnv-v0` | ``DummyEnv`` |
/// | `CartPole-v1` | ``CartPole`` (+ default wrapper stack in factory) |
/// | `Pendulum-v1` | ``Pendulum`` (+ default wrapper stack in factory) |
///
/// Example:
///
/// ```swift
/// try RLXEnvsRegistration.registerDefaults()
/// let env = try EnvironmentRegistry.shared.make("CartPole-v1")
/// ```
public enum RLXEnvsRegistration {
    /// Register DummyEnv, CartPole-v1, and Pendulum-v1 on the given registry (default: ``EnvironmentRegistry/shared``).
    public static func registerDefaults(
        on registry: EnvironmentRegistry = .shared
    ) throws {
        try registerDummyEnv(on: registry)
        try registerCartPole(on: registry)
        try registerPendulum(on: registry)
    }

    public static func registerDummyEnv(
        on registry: EnvironmentRegistry = .shared
    ) throws {
        let spec = EnvSpec(
            id: "DummyEnv-v0",
            maxEpisodeSteps: 10,
            nondeterministic: false,
            version: 1
        )
        let factory = ClosureEnvironmentFactory { config, _ in
            if config != nil {
                throw RegistryError.invalidConfig("DummyEnv-v0 ignores config; pass nil")
            }
            return AnyEnvironment(DummyEnv())
        }
        try registry.register(id: "DummyEnv-v0", spec: spec, factory: factory)
    }

    public static func registerCartPole(
        on registry: EnvironmentRegistry = .shared
    ) throws {
        let spec = EnvSpec(
            id: "CartPole-v1",
            maxEpisodeSteps: 500,
            rewardThreshold: 475,
            nondeterministic: false,
            defaultRenderMode: RenderMode.none,
            version: 1
        )
        let factory = ClosureEnvironmentFactory { config, renderMode in
            let cfg: CartPoleConfig
            if let config {
                guard let c = config as? CartPoleConfig else {
                    throw RegistryError.invalidConfig(
                        "CartPole-v1 expects CartPoleConfig, got \(type(of: config))"
                    )
                }
                cfg = c
            } else {
                cfg = .default
            }
            return CartPoleEnv.makeAny(config: cfg, renderMode: renderMode)
        }
        try registry.register(id: "CartPole-v1", spec: spec, factory: factory)
    }

    public static func registerPendulum(
        on registry: EnvironmentRegistry = .shared
    ) throws {
        let spec = EnvSpec(
            id: "Pendulum-v1",
            maxEpisodeSteps: 200,
            nondeterministic: false,
            defaultRenderMode: RenderMode.none,
            version: 1
        )
        let factory = ClosureEnvironmentFactory { config, renderMode in
            let cfg: PendulumConfig
            if let config {
                guard let c = config as? PendulumConfig else {
                    throw RegistryError.invalidConfig(
                        "Pendulum-v1 expects PendulumConfig, got \(type(of: config))"
                    )
                }
                cfg = c
            } else {
                cfg = .default
            }
            return PendulumEnv.makeAny(config: cfg, renderMode: renderMode)
        }
        try registry.register(id: "Pendulum-v1", spec: spec, factory: factory)
    }
}
