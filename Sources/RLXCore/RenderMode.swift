// RenderMode — construction-time rendering policy (design.md §17.1).

@preconcurrency import MLX

/// How an environment should produce visual/text frames.
///
/// Chosen at **construction** / `registry.make`, not per `render()` call.
public enum RenderMode: String, Sendable, Equatable, CaseIterable, Codable {
    /// No render support required; `render()` returns `nil` or throws `renderNotSupported`.
    case none
    /// Side-effect display (window / console); `render()` returns `nil` after displaying.
    case human
    /// Return an RGB frame buffer (`RenderFrame.rgb`).
    case rgbArray
    /// Return a text frame (`RenderFrame.ansi`) for toy / text envs.
    case ansi
}

/// Payload returned by `Renderable.render()` (design.md §17.2).
///
/// Introduced alongside `RenderMode` so call sites have a stable return type
/// even before concrete envs implement rendering.
///
/// `@unchecked Sendable` because `.rgb(MLXArray)` is not formally Sendable.
public enum RenderFrame: @unchecked Sendable {
    /// HxWxC image tensor (typically `UInt8` or `Float` channels).
    case rgb(MLXArray)
    /// ANSI / plain-text frame for console envs.
    case ansi(String)
    /// Human mode already displayed the frame; no payload.
    case humanDisplayed
}

extension RenderFrame: Equatable {
    public static func == (lhs: RenderFrame, rhs: RenderFrame) -> Bool {
        switch (lhs, rhs) {
        case (.ansi(let a), .ansi(let b)):
            return a == b
        case (.humanDisplayed, .humanDisplayed):
            return true
        case (.rgb(let a), .rgb(let b)):
            if a.shape != b.shape || a.dtype != b.dtype { return false }
            let elementCount = a.shape.reduce(1, *)
            if elementCount == 0 { return true }
            return Device.withDefaultDevice(.cpu) {
                eval(a, b)
                if a.dtype == .uint8 {
                    return a.asArray(UInt8.self) == b.asArray(UInt8.self)
                }
                if a.dtype == .float32 {
                    return a.asArray(Float.self) == b.asArray(Float.self)
                }
                return a === b
            }
        default:
            return false
        }
    }
}
