// Stub ALE shim when RLX_ALE_ENABLED is not defined.

#include "rlx_ale_shim.h"

#if !defined(RLX_ALE_ENABLED)

#include <cstdlib>

struct RlxAleHandle {
    int unused;
};

extern "C" int rlx_ale_is_linked(void) { return 0; }

extern "C" RlxAleHandle *rlx_ale_create(void) {
    return static_cast<RlxAleHandle *>(std::calloc(1, sizeof(RlxAleHandle)));
}

extern "C" void rlx_ale_destroy(RlxAleHandle *handle) {
    std::free(handle);
}

extern "C" RlxAleStatus rlx_ale_set_int(RlxAleHandle *, const char *, int) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" RlxAleStatus rlx_ale_set_float(RlxAleHandle *, const char *, float) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" RlxAleStatus rlx_ale_set_bool(RlxAleHandle *, const char *, int) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" RlxAleStatus rlx_ale_load_rom(RlxAleHandle *, const char *) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" RlxAleStatus rlx_ale_reset(RlxAleHandle *) { return RLX_ALE_ERR_NO_ALE; }
extern "C" int rlx_ale_screen_height(const RlxAleHandle *) { return 0; }
extern "C" int rlx_ale_screen_width(const RlxAleHandle *) { return 0; }
extern "C" int rlx_ale_minimal_action_count(const RlxAleHandle *) { return 0; }
extern "C" RlxAleStatus rlx_ale_act_minimal_index(RlxAleHandle *, int, float *) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" int rlx_ale_game_over(const RlxAleHandle *) { return 1; }
extern "C" int rlx_ale_lives(const RlxAleHandle *) { return 0; }
extern "C" RlxAleStatus rlx_ale_copy_screen_gray(const RlxAleHandle *, unsigned char *, int) {
    return RLX_ALE_ERR_NO_ALE;
}
extern "C" RlxAleStatus rlx_ale_copy_screen_rgb(const RlxAleHandle *, unsigned char *, int) {
    return RLX_ALE_ERR_NO_ALE;
}

#endif /* !RLX_ALE_ENABLED */
