#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_fast_01_block_sums(
    __global const uint* input,
    __global       uint* block_sums)
{
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint base = group * SCAN_BLOCK;

    uint sum = 0;

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        sum += input[base + lid + j * GROUP_SIZE];
    }

    __local uint local_sums[GROUP_SIZE];

    local_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            local_sums[lid] += local_sums[lid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        block_sums[group] = local_sums[0];
    }
}
