// RLXCore — reinforcement learning environment & data-collection substrate on mlx-swift.
//
// Core contracts: Environment, Space, results, Seed/PRNG, registry, errors, type erasure.
// Normative design: repository design.md. Developer map: Documentation/DeveloperGuide.md.

import MLX

/// Module identity and lightweight diagnostics for the **RLXCore** product.
///
/// `RLXCore` is the foundation target of `rlx-swift`: MDP protocols, spaces, seeding,
/// registry, and type erasure. It depends on **MLX** only. Higher layers (`RLXWrappers`,
/// `RLXEnvs`, `RLXVector`, `RLXTesting`) build on this module.
///
/// - See also: DocC *Architecture for developers*; repository `Documentation/DeveloperGuide.md`.
public enum RLXCore {
    /// Package version string for this source line (`-dev` means unreleased).
    ///
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
