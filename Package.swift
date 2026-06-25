// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Aligned with the pinned mlx-swift release (see dependencies below).

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
            dependencies: ["RLXCore"],
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
