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
        .testTarget(
            name: "RLXCoreTests",
            dependencies: [
                "RLXCore",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/RLXCoreTests"
        ),
        // Local/CI-independent smoke executable (no XCTest / Metal runtime required).
        .executableTarget(
            name: "RLXCoreSmoke",
            dependencies: ["RLXCore"],
            path: "Sources/RLXCoreSmoke"
        ),
    ]
)
