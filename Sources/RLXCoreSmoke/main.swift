// PR-01 scaffold smoke checks — runnable without XCTest / Metal runtime resources.
// Confirms RLXCore product links and exports identity. MLX runtime eval is covered
// under full Xcode (CI tier-1 `swift test` with Metal library resources).

import Foundation
import RLXCore

enum SmokeFailure: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}

func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw SmokeFailure.message(message) }
}

do {
    try expect(!RLXCore.version.isEmpty, "RLXCore.version must be non-empty")
    try expect(RLXCore.version.contains("0.1.0"), "RLXCore.version should contain 0.1.0, got \(RLXCore.version)")
    // Function reference only — do not invoke (avoids MLXArray/Metal init on CLI toolchains).
    let _: () -> Any = { RLXCore.mlxSmokeArray() as Any }

    print("RLXCoreSmoke: all checks passed (rlx-swift \(RLXCore.version), RLXCore+MLX linked at build time)")
    exit(0)
} catch {
    fputs("RLXCoreSmoke FAILED: \(error)\n", stderr)
    exit(1)
}
