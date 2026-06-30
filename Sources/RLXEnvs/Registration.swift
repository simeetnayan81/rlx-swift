// Registration — default factories into EnvironmentRegistry (design.md §18.1).

import RLXCore

/// Registers built-in toy / debug envs. Safe to call once; second call throws ``RegistryError/duplicateID``.
public enum RLXEnvsRegistration {
    /// Register ``DummyEnv`` as `"DummyEnv-v0"` on ``EnvironmentRegistry/shared``.
    public static func registerDefaults(
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
}
