#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts,
    __global       uint* block_sums,
    uint m)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint local_counts[GROUP_SIZE];
    local_counts[local_id] = (id < m) ? counts[id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint num_count_blocks = GROUP_SIZE / RADIX_SIZE;
    for (uint stride = num_count_blocks / 2; stride > 0; stride >>= 1) {
        if (local_id / RADIX_SIZE < stride) {
            local_counts[local_id] += local_counts[local_id + stride * RADIX_SIZE];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id < RADIX_SIZE) {
        block_sums[group_id * RADIX_SIZE + local_id] = local_counts[local_id];
    }
}
