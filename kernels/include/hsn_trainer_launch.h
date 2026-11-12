/* P2.10b trainer kernels: the private launch descriptor of kernels/cuda/hsn_trainer.cu.
 * Passed by value to every exported kernel. A launch walks a list of tensors as one index space:
 * tensor j holds the elements [end[j - 1], end[j]) (end[-1] = 0) and p[r][j] is the device address
 * of its r-th operand in the caller's current CUDA context (each kernel documents its rows; a row
 * a kernel does not read is left null). i[] are sizes and flags (i[0] is always the number of
 * tensors), f[] float scalars, g[] double scalars, u[c][m] per-unit coefficients of the stacked
 * unit tensors. Not part of the public numerical ABI (hsn_ops.h) and not loaded by the Rust
 * backend. The struct stays under the 4 KiB a kernel's arguments may take.
 */
#ifndef HSN_TRAINER_LAUNCH_H
#define HSN_TRAINER_LAUNCH_H
#include <stdint.h>
#define HSN_TR_ROWS 4
#define HSN_TR_TENSORS 40
#define HSN_TR_INTS 8
#define HSN_TR_FLOATS 8
#define HSN_TR_DOUBLES 2
#define HSN_TR_COEFS 6
#define HSN_TR_UNITS 64
struct HsnTrainArgs {
    void* p[HSN_TR_ROWS][HSN_TR_TENSORS];
    int64_t end[HSN_TR_TENSORS];
    double g[HSN_TR_DOUBLES];
    int32_t i[HSN_TR_INTS];
    float f[HSN_TR_FLOATS];
    float u[HSN_TR_COEFS][HSN_TR_UNITS];
};
#ifdef __cplusplus
static_assert(sizeof(void*) == 8, "HSN trainer kernels require a 64-bit platform");
static_assert(sizeof(HsnTrainArgs) == 8 * HSN_TR_ROWS * HSN_TR_TENSORS + 8 * HSN_TR_TENSORS
    + 8 * HSN_TR_DOUBLES + 4 * HSN_TR_INTS + 4 * HSN_TR_FLOATS + 4 * HSN_TR_COEFS * HSN_TR_UNITS,
    "trainer launch descriptor size changed");
static_assert(sizeof(HsnTrainArgs) <= 4096, "a kernel's arguments may take at most 4 KiB");
#endif
#endif
