// Real ALE C++ shim (active only with -DRLX_ALE_ENABLED).

#include "rlx_ale_shim.h"

#if defined(RLX_ALE_ENABLED)

#if defined(__has_include)
#  if __has_include(<ale/ale_interface.hpp>)
#    include <ale/ale_interface.hpp>
#  elif __has_include(<ale_interface.hpp>)
#    include <ale_interface.hpp>
#  else
#    error "ALE headers not found (ale/ale_interface.hpp or ale_interface.hpp)"
#  endif
#else
#  include <ale_interface.hpp>
#endif

#include <cstring>
#include <memory>
#include <vector>

struct RlxAleHandle {
    std::unique_ptr<ale::ALEInterface> ale;
    ale::ActionVect minimal;
    bool loaded = false;
};

extern "C" int rlx_ale_is_linked(void) { return 1; }

extern "C" RlxAleHandle *rlx_ale_create(void) {
    try {
        auto *h = new RlxAleHandle();
        h->ale = std::make_unique<ale::ALEInterface>();
        return h;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void rlx_ale_destroy(RlxAleHandle *handle) { delete handle; }

extern "C" RlxAleStatus rlx_ale_set_int(RlxAleHandle *handle, const char *key, int value) {
    if (!handle || !handle->ale || !key) return RLX_ALE_ERR_ARG;
    try {
        handle->ale->setInt(key, value);
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" RlxAleStatus rlx_ale_set_float(RlxAleHandle *handle, const char *key, float value) {
    if (!handle || !handle->ale || !key) return RLX_ALE_ERR_ARG;
    try {
        handle->ale->setFloat(key, value);
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" RlxAleStatus rlx_ale_set_bool(RlxAleHandle *handle, const char *key, int value) {
    if (!handle || !handle->ale || !key) return RLX_ALE_ERR_ARG;
    try {
        handle->ale->setBool(key, value != 0);
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" RlxAleStatus rlx_ale_load_rom(RlxAleHandle *handle, const char *path) {
    if (!handle || !handle->ale || !path) return RLX_ALE_ERR_ARG;
    try {
        handle->ale->loadROM(path);
        handle->minimal = handle->ale->getMinimalActionSet();
        handle->loaded = true;
        return RLX_ALE_OK;
    } catch (...) {
        handle->loaded = false;
        return RLX_ALE_ERR_ROM;
    }
}

extern "C" RlxAleStatus rlx_ale_reset(RlxAleHandle *handle) {
    if (!handle || !handle->ale || !handle->loaded) return RLX_ALE_ERR_STATE;
    try {
        handle->ale->reset_game();
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" int rlx_ale_screen_height(const RlxAleHandle *handle) {
    if (!handle || !handle->ale || !handle->loaded) return 0;
    return static_cast<int>(handle->ale->getScreen().height());
}

extern "C" int rlx_ale_screen_width(const RlxAleHandle *handle) {
    if (!handle || !handle->ale || !handle->loaded) return 0;
    return static_cast<int>(handle->ale->getScreen().width());
}

extern "C" int rlx_ale_minimal_action_count(const RlxAleHandle *handle) {
    if (!handle || !handle->loaded) return 0;
    return static_cast<int>(handle->minimal.size());
}

extern "C" RlxAleStatus rlx_ale_act_minimal_index(RlxAleHandle *handle, int index, float *reward_out) {
    if (!handle || !handle->ale || !handle->loaded || !reward_out) return RLX_ALE_ERR_ARG;
    if (index < 0 || index >= static_cast<int>(handle->minimal.size())) return RLX_ALE_ERR_ARG;
    try {
        *reward_out = static_cast<float>(handle->ale->act(handle->minimal[static_cast<size_t>(index)]));
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" int rlx_ale_game_over(const RlxAleHandle *handle) {
    if (!handle || !handle->ale || !handle->loaded) return 1;
    return handle->ale->game_over() ? 1 : 0;
}

extern "C" int rlx_ale_lives(const RlxAleHandle *handle) {
    if (!handle || !handle->ale || !handle->loaded) return 0;
    return handle->ale->lives();
}

extern "C" RlxAleStatus rlx_ale_copy_screen_gray(
    const RlxAleHandle *handle,
    unsigned char *buffer,
    int buffer_len
) {
    if (!handle || !handle->ale || !handle->loaded || !buffer) return RLX_ALE_ERR_ARG;
    const int h = rlx_ale_screen_height(handle);
    const int w = rlx_ale_screen_width(handle);
    const int need = h * w;
    if (need <= 0 || buffer_len < need) return RLX_ALE_ERR_BUFFER;
    try {
        std::vector<unsigned char> gray(static_cast<size_t>(need));
        handle->ale->getScreenGrayscale(gray);
        std::memcpy(buffer, gray.data(), static_cast<size_t>(need));
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

extern "C" RlxAleStatus rlx_ale_copy_screen_rgb(
    const RlxAleHandle *handle,
    unsigned char *buffer,
    int buffer_len
) {
    if (!handle || !handle->ale || !handle->loaded || !buffer) return RLX_ALE_ERR_ARG;
    const int h = rlx_ale_screen_height(handle);
    const int w = rlx_ale_screen_width(handle);
    const int need = h * w * 3;
    if (need <= 0 || buffer_len < need) return RLX_ALE_ERR_BUFFER;
    try {
        std::vector<unsigned char> rgb(static_cast<size_t>(need));
        handle->ale->getScreenRGB(rgb);
        std::memcpy(buffer, rgb.data(), static_cast<size_t>(need));
        return RLX_ALE_OK;
    } catch (...) {
        return RLX_ALE_ERR_INTERNAL;
    }
}

#endif /* RLX_ALE_ENABLED */
