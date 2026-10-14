#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_03_write_result(
    __global const uint* input,
    __global const uint* tree,
    __global       uint* prefix_sum_accum, // результат: prefix_sum_accum[i] = sum[0..i]
    unsigned int pairs_off,                // смещение нижнего уровня в tree
    unsigned int n)
{
    const size_t idx = get_global_id(0);

    if (idx >= (n + 1) / 2) {
        return;
    }

    const uint p = tree[pairs_off + idx];
    const uint x = p + input[2 * idx];
    prefix_sum_accum[2 * idx] = x;
    if (2 * idx + 1 < n) {
        prefix_sum_accum[2 * idx + 1] = x + input[2 * idx + 1];
    }
}
