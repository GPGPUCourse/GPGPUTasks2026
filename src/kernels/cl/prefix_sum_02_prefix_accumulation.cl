#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* block_prefix_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    size_t block = get_group_id(0);
    if (block == 0) {
        return;
    }

    uint prefix = block_prefix_sum[block - 1];
    size_t first = block * (2 * GROUP_SIZE) + get_local_id(0);
    size_t second = first + GROUP_SIZE;
    if (first < n) {
        prefix_sum_accum[first] += prefix;
    }
    if (second < n) {
        prefix_sum_accum[second] += prefix;
    }
}
