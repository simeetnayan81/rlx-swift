import RLXCore
import XCTest

/// Tiny config for invalidConfig tests.
private struct SmokeConfig: EnvConfig {
    var n: Int
}

final class RegistryTests: XCTestCase {

    func testRegisterMakeAndSpec() throws {
        let reg = EnvironmentRegistry()
        let spec = EnvSpec(id: "Toy-v0", maxEpisodeSteps: 2, version: 1)
        let factory = ClosureEnvironmentFactory { config, renderMode in
            XCTAssertNil(config)
            XCTAssertEqual(renderMode, .none)
            // Minimal stand-in: type-erase a throwaway using a private env would need envs module;
            // use a factory that builds AnyEnvironment from a tiny class defined below.
            return AnyEnvironment(RegistryProbeEnv())
        }
        try reg.register(id: "Toy-v0", spec: spec, factory: factory)
        XCTAssertEqual(reg.ids, ["Toy-v0"])
        XCTAssertEqual(reg.spec(for: "Toy-v0")?.maxEpisodeSteps, 2)

        let env = try reg.make("Toy-v0")
        let r = try env.reset()
        XCTAssertEqual(r.observation as? Int, 0)
        try env.close()
    }

    func testUnknownID() {
        let reg = EnvironmentRegistry()
        XCTAssertThrowsError(try reg.make("missing")) { err in
            XCTAssertEqual(err as? RegistryError, .unknownID("missing"))
        }
    }

    func testDuplicateID() throws {
        let reg = EnvironmentRegistry()
        let factory = ClosureEnvironmentFactory { _, _ in AnyEnvironment(RegistryProbeEnv()) }
        try reg.register(id: "X", spec: EnvSpec(id: "X"), factory: factory)
        XCTAssertThrowsError(
            try reg.register(id: "X", spec: EnvSpec(id: "X"), factory: factory)
        ) { err in
            XCTAssertEqual(err as? RegistryError, .duplicateID("X"))
        }
    }

    func testInvalidConfigPropagates() throws {
        let reg = EnvironmentRegistry()
        let factory = ClosureEnvironmentFactory { config, _ in
            guard config == nil else {
                throw RegistryError.invalidConfig("expected nil")
            }
            return AnyEnvironment(RegistryProbeEnv())
        }
        try reg.register(id: "Y", spec: EnvSpec(id: "Y"), factory: factory)
        XCTAssertThrowsError(try reg.make("Y", config: SmokeConfig(n: 1))) { err in
            XCTAssertEqual(err as? RegistryError, .invalidConfig("expected nil"))
        }
        _ = try reg.make("Y", config: nil)
    }

    func testRegisterAlignsSpecId() throws {
        let reg = EnvironmentRegistry()
        try reg.register(
            id: "Align-v0",
            spec: EnvSpec(id: "wrong"),
            factory: ClosureEnvironmentFactory { _, _ in AnyEnvironment(RegistryProbeEnv()) }
        )
        XCTAssertEqual(reg.spec(for: "Align-v0")?.id, "Align-v0")
    }

    func testIdsSorted() throws {
        let reg = EnvironmentRegistry()
        let f = ClosureEnvironmentFactory { _, _ in AnyEnvironment(RegistryProbeEnv()) }
        try reg.register(id: "b", spec: EnvSpec(id: "b"), factory: f)
        try reg.register(id: "a", spec: EnvSpec(id: "a"), factory: f)
        XCTAssertEqual(reg.ids, ["a", "b"])
    }

    func testRegistryErrorDescriptions() {
        XCTAssertTrue(RegistryError.unknownID("x").description.contains("unknownID"))
        XCTAssertTrue(RegistryError.duplicateID("x").description.contains("duplicateID"))
    }
}

/// Private probe env for registry tests (Int discrete).
private final class RegistryProbeEnv: Environment {
    typealias Observation = Int
    typealias Action = Int
    typealias ObservationSpace = DiscreteSpace
    typealias ActionSpace = DiscreteSpace

    let observationSpace = DiscreteSpace(n: 2)
    let actionSpace = DiscreteSpace(n: 2)
    private var closed = false

    func reset(seed: UInt64?, options: (any ResetOptions)?) throws -> ResetResult<Int> {
        if closed { throw EnvironmentError.closed }
        return ResetResult(observation: 0)
    }

    func step(_ action: Int) throws -> StepResult<Int> {
        if closed { throw EnvironmentError.closed }
        return StepResult(observation: action, reward: 0, terminated: true, truncated: false)
    }

    func close() throws { closed = true }
}
