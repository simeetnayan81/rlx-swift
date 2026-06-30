// RLXCore — reinforcement learning environment & data-collection substrate on mlx-swift.
//
// Hosts result types, errors, Seed/PRNG helpers, and (incrementally)
// Environment/Space protocols and registry. See design.md §28 PR plan.

import MLX

/// Package identity marker for tests and diagnostics.
public enum RLXCore {
    /// Development version of the package on this branch/line (`-dev` = unreleased).
    /// Release tags set this to a bare SemVer (e.g. `"0.1.0"`) on the `release` branch only.
    public static let version = "0.2.0-dev"

    /// Confirms the MLX product is linked and usable from RLXCore.
    ///
    /// Uses the current default `Device` (CPU or GPU). Callers that run outside
    /// an Xcode/app bundle may need `Device.setDefault(device: .cpu)` first if
    /// the Metal default library is not available.
    public static func mlxSmokeArray() -> MLXArray {
        MLXArray(Float(1.0))
    }
}
