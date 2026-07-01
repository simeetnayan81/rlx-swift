/// Whether this binary was linked against a real ALE install (`ALE_ROOT` at build time).
public enum RLXALE {
    /// `true` if the C++ shim was compiled with `RLX_ALE_ENABLED`.
    public static var isALELinked: Bool { ALEBridge.isLinked() }
}
