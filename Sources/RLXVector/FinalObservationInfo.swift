import MLX
import RLXCore
import RLXWrappers

enum FinalObservationInfo {
    /// Store a type-erased observation under ``InfoKeys/finalObservation`` when possible.
    static func embed(_ observation: Any, into info: inout Info) {
        if let i = observation as? Int {
            info[InfoKeys.finalObservation] = .int(i)
        } else if let a = observation as? MLXArray {
            info[InfoKeys.finalObservation] = .array(a)
        } else if let f = observation as? Float {
            info[InfoKeys.finalObservation] = .double(Double(f))
        } else if let d = observation as? Double {
            info[InfoKeys.finalObservation] = .double(d)
        } else if let b = observation as? Bool {
            info[InfoKeys.finalObservation] = .bool(b)
        } else if let s = observation as? String {
            info[InfoKeys.finalObservation] = .string(s)
        }
        // Unsupported types: leave key absent; rewards/flags still valid.
    }
}
