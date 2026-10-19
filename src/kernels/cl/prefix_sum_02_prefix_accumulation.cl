#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* values,
    __global const uint* pair_prefix,
    __global       uint* prefix_sum_accum,
    unsigned int n
) {
    const size_t i = get_global_id(0);
    const size_t left = 2 * i;
    if (left >= n) {
        return;
    }
    const uint prefix = pair_prefix[i];
    if (left + 1 < n) {
        prefix_sum_accum[left] = prefix - values[left + 1];
        prefix_sum_accum[left + 1] = prefix;
    } else {
        prefix_sum_accum[left] = prefix;
    }
}
