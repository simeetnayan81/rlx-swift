import RLXCore
import XCTest

/// Smoke tests for PR-01 package scaffold: targets link and package identity is present.
///
/// Full MLX `eval` runtime checks require Metal library resources shipped with mlx-swift
/// under a full Xcode / app-bundle environment. Those belong in later PRs once env logic exists.
final class PackageScaffoldTests: XCTestCase {

    func testRLXCoreExportsVersionString() {
        XCTAssertFalse(RLXCore.version.isEmpty)
        XCTAssertTrue(RLXCore.version.contains("0.1.0"))
    }

    func testRLXCoreExposesMLXSmokeFactory() {
        // Compile-time linkage: the factory is public and returns MLXArray (imported via RLXCore/MLX).
        // Do not call it here — constructing MLXArray initializes Metal and needs default.metallib.
        let factory = RLXCore.mlxSmokeArray
        XCTAssertNotNil(factory)
    }
}
