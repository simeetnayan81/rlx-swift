import RLXCore
import RLXEnvs
import XCTest

final class RegistrationTests: XCTestCase {

    func testRegisterDefaultsMakeDummyEnv() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerDefaults(on: reg)
        XCTAssertTrue(reg.ids.contains("DummyEnv-v0"))
        XCTAssertEqual(reg.spec(for: "DummyEnv-v0")?.id, "DummyEnv-v0")

        let env = try reg.make("DummyEnv-v0")
        let r = try env.reset(seed: Seed(1))
        XCTAssertEqual(r.observation as? Int, 0)
        let s = try env.step(1)
        XCTAssertEqual(s.observation as? Int, 1)
        try env.close()
    }

    func testRegisterDefaultsRejectsConfig() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerDefaults(on: reg)
        struct C: EnvConfig {}
        XCTAssertThrowsError(try reg.make("DummyEnv-v0", config: C())) { err in
            guard case .invalidConfig = err as? RegistryError else {
                return XCTFail("expected invalidConfig, got \(err)")
            }
        }
    }

    func testRegisterDefaultsDuplicateOnSameRegistry() throws {
        let reg = EnvironmentRegistry()
        try RLXEnvsRegistration.registerDefaults(on: reg)
        XCTAssertThrowsError(try RLXEnvsRegistration.registerDefaults(on: reg)) { err in
            XCTAssertEqual(err as? RegistryError, .duplicateID("DummyEnv-v0"))
        }
    }
}
