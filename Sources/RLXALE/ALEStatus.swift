// Maps C shim status codes to EnvironmentError.

import RLXALECXX
import RLXCore

enum ALEBridge {
    static func isLinked() -> Bool {
        rlx_ale_is_linked() != 0
    }

    static func check(_ status: RlxAleStatus, context: String) throws {
        switch status {
        case RLX_ALE_OK:
            return
        case RLX_ALE_ERR_NO_ALE:
            throw EnvironmentError.configuration(
                "ALE C++ library not linked. Build with ALE_ROOT set to an ALE install prefix (see docs/ale-adapter-design.md)."
            )
        case RLX_ALE_ERR_ROM:
            throw EnvironmentError.configuration("\(context): failed to load ROM")
        case RLX_ALE_ERR_STATE:
            throw EnvironmentError.configuration("\(context): invalid ALE state (load ROM / reset first)")
        case RLX_ALE_ERR_ARG:
            throw EnvironmentError.invalidAction("\(context): invalid argument")
        case RLX_ALE_ERR_BUFFER:
            throw EnvironmentError.configuration("\(context): screen buffer too small")
        case RLX_ALE_ERR_INTERNAL:
            throw EnvironmentError.underlying(ALEInternalError(message: context))
        default:
            throw EnvironmentError.configuration("\(context): unknown ALE status \(status.rawValue)")
        }
    }
}

struct ALEInternalError: Error, CustomStringConvertible {
    let message: String
    var description: String { "ALE internal error: \(message)" }
}
