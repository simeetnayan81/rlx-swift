// swift-tools-version: 6.0
// Minimum SwiftPM tools version for this package.
// Pinned mlx-swift 0.31.4 declares 5.12; we use 6.0 for Swift 6 language mode.
// (mlx-swift main may require 6.3 — do not copy main's tools-version unless you bump the pin.)
//
// Optional ALE (Atari): set ALE_ROOT to an ALE CMake install prefix, then rebuild.
// See docs/ale-adapter-design.md.

import PackageDescription

// When set to an ALE install prefix (include/ + lib/), link the real C++ library.
let aleRoot = Context.environment["ALE_ROOT"]
let aleEnabled = !(aleRoot ?? "").isEmpty

var aleCXXSettings: [CXXSetting] = [
    .headerSearchPath("include"),
]
var aleLinkerSettings: [LinkerSetting] = []

if aleEnabled, let root = aleRoot {
    aleCXXSettings.append(contentsOf: [
        .define("RLX_ALE_ENABLED"),
        .unsafeFlags([
            "-I\(root)/include",
            "-I\(root)/include/ale",
        ]),
    ])
    aleLinkerSettings.append(contentsOf: [
        .unsafeFlags([
            "-L\(root)/lib",
            "-L\(root)/lib64",
            "-lale",
            // Common ALE transitive deps (may be unused if static/shared already embeds):
            "-lz",
        ]),
    ])
}

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
        .library(
            name: "RLXVector",
            targets: ["RLXVector"]
        ),
        // Optional Atari adapter (works as stub without ALE_ROOT; real ALE when ALE_ROOT is set).
        .library(
            name: "RLXALE",
            targets: ["RLXALE"]
        ),
        .executable(
            name: "RandomAgentDemo",
            targets: ["RandomAgentDemo"]
        ),
        .executable(
            name: "ALERandomAgent",
            targets: ["ALERandomAgent"]
        ),
        .executable(
            name: "ALEGifDemo",
            targets: ["ALEGifDemo"]
        ),
    ],
    dependencies: [
        // Pin mlx-swift; platforms/tools-version inherit from this pin (design.md §27.2).
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        // DocC generation: `swift package generate-documentation --target RLXCore` (etc.)
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.3.0"),
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
        .target(
            name: "RLXVector",
            dependencies: [
                "RLXCore",
                "RLXWrappers",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/RLXVector"
        ),
        // C++ shim over ALE (stub if ALE_ROOT unset).
        .target(
            name: "RLXALECXX",
            path: "Sources/RLXALECXX",
            publicHeadersPath: "include",
            cxxSettings: aleCXXSettings,
            linkerSettings: aleLinkerSettings
        ),
        .target(
            name: "RLXALE",
            dependencies: [
                "RLXCore",
                "RLXALECXX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/RLXALE"
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
        .testTarget(
            name: "RLXVectorTests",
            dependencies: ["RLXVector", "RLXEnvs", "RLXCore", "RLXWrappers"],
            path: "Tests/RLXVectorTests"
        ),
        .testTarget(
            name: "RLXALETests",
            dependencies: ["RLXALE", "RLXCore"],
            path: "Tests/RLXALETests"
        ),
        .executableTarget(
            name: "RLXCoreSmoke",
            dependencies: ["RLXCore", "RLXEnvs", "RLXWrappers", "RLXTesting", "RLXVector"],
            path: "Sources/RLXCoreSmoke"
        ),
        .executableTarget(
            name: "RandomAgentDemo",
            dependencies: ["RLXCore", "RLXEnvs", "RLXWrappers"],
            path: "Examples/RandomAgentDemo"
        ),
        .executableTarget(
            name: "ALERandomAgent",
            dependencies: ["RLXALE", "RLXCore", "RLXWrappers"],
            path: "Examples/ALERandomAgent"
        ),
        .executableTarget(
            name: "ALEGifDemo",
            dependencies: ["RLXALE", "RLXCore"],
            path: "Examples/ALEGifDemo",
            exclude: ["output"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
