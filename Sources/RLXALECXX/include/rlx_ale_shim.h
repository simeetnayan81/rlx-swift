// C façade over Farama ALE C++ (docs/ale-adapter-design.md).
// When built without RLX_ALE_ENABLED, all calls return RLX_ALE_ERR_NO_ALE.

#ifndef RLX_ALE_SHIM_H
#define RLX_ALE_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RlxAleHandle RlxAleHandle;

typedef enum RlxAleStatus {
    RLX_ALE_OK = 0,
    RLX_ALE_ERR_NO_ALE = 1,
    RLX_ALE_ERR_ROM = 2,
    RLX_ALE_ERR_STATE = 3,
    RLX_ALE_ERR_ARG = 4,
    RLX_ALE_ERR_INTERNAL = 5,
    RLX_ALE_ERR_BUFFER = 6,
} RlxAleStatus;

/// 1 if compiled with real ALE, 0 if stub.
int rlx_ale_is_linked(void);

RlxAleHandle *rlx_ale_create(void);
void rlx_ale_destroy(RlxAleHandle *handle);

/// Must be called before load_rom for seed/frameskip/etc. (ALE applies settings at load).
RlxAleStatus rlx_ale_set_int(RlxAleHandle *handle, const char *key, int value);
RlxAleStatus rlx_ale_set_float(RlxAleHandle *handle, const char *key, float value);
RlxAleStatus rlx_ale_set_bool(RlxAleHandle *handle, const char *key, int value);

RlxAleStatus rlx_ale_load_rom(RlxAleHandle *handle, const char *path);
RlxAleStatus rlx_ale_reset(RlxAleHandle *handle);

int rlx_ale_screen_height(const RlxAleHandle *handle);
int rlx_ale_screen_width(const RlxAleHandle *handle);

/// Minimal action set size (0 if not loaded).
int rlx_ale_minimal_action_count(const RlxAleHandle *handle);

/// Act with index into the minimal action set. Writes reward; may leave game_over.
RlxAleStatus rlx_ale_act_minimal_index(RlxAleHandle *handle, int index, float *reward_out);

int rlx_ale_game_over(const RlxAleHandle *handle);
int rlx_ale_lives(const RlxAleHandle *handle);

/// Copy grayscale screen (height * width bytes) into buffer.
RlxAleStatus rlx_ale_copy_screen_gray(
    const RlxAleHandle *handle,
    unsigned char *buffer,
    int buffer_len
);

/// Copy RGB screen (height * width * 3 bytes, interleaved RGB) into buffer.
RlxAleStatus rlx_ale_copy_screen_rgb(
    const RlxAleHandle *handle,
    unsigned char *buffer,
    int buffer_len
);

#ifdef __cplusplus
}
#endif

#endif /* RLX_ALE_SHIM_H */
