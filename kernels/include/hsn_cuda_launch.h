/* Private device-launch ABI. The public numerical ABI remains hsn_ops.h.
 * Passed by value to each exported kernel; shared by Python's host and Rust.
 * p[] follows the argument order documented at each device implementation.
 * All pointers are device addresses in the caller's current CUDA context.
 */
#ifndef HSN_CUDA_LAUNCH_H
#define HSN_CUDA_LAUNCH_H
#include <stdint.h>
struct HsnCudaArgs {
    void* p[16];
    int32_t B, N, d, K, P, M, H, E;
};
#ifdef __cplusplus
static_assert(sizeof(void*) == 8, "HSN CUDA hosts require a 64-bit platform");
static_assert(sizeof(HsnCudaArgs) == 160, "CUDA launch ABI size changed");
#endif
#endif
