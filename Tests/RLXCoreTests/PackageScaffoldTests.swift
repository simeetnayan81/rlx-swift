import MLX
import RLXCore
import XCTest

/// Smoke tests for PR-01 package scaffold.
///
/// Run with `xcodebuild` (not plain `swift test`) so Cmlx Metal shaders are built:
///   xcodebuild test -scheme rlx-swift-Package -destination 'platform=macOS'
final class PackageScaffoldTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Prefer CPU for deterministic unit tests; still exercises MLX runtime.
        _ = Device.withDefaultDevice(.cpu) { () -> Void in }
    }

    func testRLXCoreExportsVersionString() {
        XCTAssertFalse(RLXCore.version.isEmpty)
        XCTAssertTrue(RLXCore.version.contains("0.2.0"))
    }

    func testRLXCoreCanConstructAndEvalMLXArray() {
        Device.withDefaultDevice(.cpu) {
            let array = RLXCore.mlxSmokeArray()
            eval(array)
            XCTAssertEqual(array.shape, [])
            XCTAssertEqual(array.item(Float.self), 1.0)
        }
    }

    func testMLXIsImportableAndOperableFromTestTarget() {
        Device.withDefaultDevice(.cpu) {
            let a = MLXArray([1.0, 2.0, 3.0] as [Float])
            let b = a + a
            eval(b)
            XCTAssertEqual(b.shape, [3])
            XCTAssertEqual(b[0].item(Float.self), 2.0)
        }
    }
}
