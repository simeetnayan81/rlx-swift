// RLXCore — reinforcement learning environment & data-collection substrate on mlx-swift.
//
// This module will host Environment/Space protocols, result types, seeding, and errors.
// Behaviour is added incrementally (see design.md §28 PR plan); PR-01 is scaffold only.

import MLX

/// Package identity marker for tests and diagnostics.
public enum RLXCore {
    /// Semantic version of the `rlx-swift` package (scaffold placeholder).
    public static let version = "0.1.0-dev"

    /// Confirms the MLX product is linked and usable from RLXCore.
    ///
    /// Uses the current default `Device` (CPU or GPU). Callers that run outside
    /// an Xcode/app bundle may need `Device.setDefault(device: .cpu)` first if
    /// the Metal default library is not available.
    public static func mlxSmokeArray() -> MLXArray {
        MLXArray(Float(1.0))
    }
}
