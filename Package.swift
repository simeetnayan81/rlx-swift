// swift-tools-version: 6.0
// Minimum SwiftPM tools version for this package.
// Pinned mlx-swift 0.31.4 declares 5.12; we use 6.0 for Swift 6 language mode.
// (mlx-swift main may require 6.3 — do not copy main's tools-version unless you bump the pin.)

import PackageDescription

let package = Package(
    name: "rlx-swift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "RLXCore",
            targets: ["RLXCore"]
        ),
        .library(
            name: "RLXWrappers",
            targets: ["RLXWrappers"]
        ),
        .library(
            name: "RLXEnvs",
            targets: ["RLXEnvs"]
        ),
        .library(
            name: "RLXTesting",
            targets: ["RLXTesting"]
        ),
    ],
    dependencies: [
        // Pin mlx-swift; platforms/tools-version inherit from this pin (design.md §27.2).
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
    ],
    targets: [
        .target(
            name: "RLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/RLXCore"
        ),
        .target(
            name: "RLXWrappers",
            dependencies: ["RLXCore"],
            path: "Sources/RLXWrappers"
        ),
        .target(
            name: "RLXEnvs",
            dependencies: ["RLXCore", "RLXWrappers"],
            path: "Sources/RLXEnvs"
        ),
        .target(
            name: "RLXTesting",
            dependencies: ["RLXCore", "RLXWrappers"],
            path: "Sources/RLXTesting"
        ),
        .testTarget(
            name: "RLXCoreTests",
            dependencies: [
                "RLXCore",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/RLXCoreTests"
        ),
        .testTarget(
            name: "RLXWrappersTests",
            dependencies: ["RLXWrappers", "RLXEnvs"],
            path: "Tests/RLXWrappersTests"
        ),
        .testTarget(
            name: "RLXEnvsTests",
            dependencies: ["RLXEnvs", "RLXCore", "RLXWrappers"],
            path: "Tests/RLXEnvsTests"
        ),
        .testTarget(
            name: "RLXTestingTests",
            dependencies: ["RLXTesting", "RLXEnvs", "RLXWrappers"],
            path: "Tests/RLXTestingTests"
        ),
        // CLI / Linux smoke executable (no XCTest). Links core, envs, wrappers, and testing
        // helpers on Discrete / pure-Swift paths; MLXArray-heavy checks stay in XCTest.
        .executableTarget(
            name: "RLXCoreSmoke",
            dependencies: ["RLXCore", "RLXEnvs", "RLXWrappers", "RLXTesting"],
            path: "Sources/RLXCoreSmoke"
        ),
    ]
)
