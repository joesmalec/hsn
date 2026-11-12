/* P2.10 window kernels: the private launch descriptor of kernels/cuda/hsn_window.cu.
 * Passed by value to every exported kernel. p[] holds device addresses in the caller's current
 * CUDA context, i[] sizes / flags / the position index, f[] scalars; each kernel documents its
 * layout. Not part of the public numerical ABI (hsn_ops.h) and not loaded by the Rust backend.
 */
#ifndef HSN_WINDOW_LAUNCH_H
#define HSN_WINDOW_LAUNCH_H
#include <stdint.h>
#define HSN_WIN_POINTERS 48
#define HSN_WIN_INTS 32
#define HSN_WIN_FLOATS 8
struct HsnWinArgs {
    void* p[HSN_WIN_POINTERS];
    int32_t i[HSN_WIN_INTS];
    float f[HSN_WIN_FLOATS];
};
#ifdef __cplusplus
static_assert(sizeof(void*) == 8, "HSN window kernels require a 64-bit platform");
static_assert(sizeof(HsnWinArgs) == 8 * HSN_WIN_POINTERS + 4 * HSN_WIN_INTS + 4 * HSN_WIN_FLOATS,
    "window launch descriptor size changed");
#endif
#endif
