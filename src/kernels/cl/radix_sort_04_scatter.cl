#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* src,
    __global const uint* offsets,
    __global       uint* dst,
    uint n,
    uint shift)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint digit_prefix_counts[GROUP_SIZE][RADIX_SIZE];

    uint value = 0u;
    uint digit = 0u;
    const bool in_bounds = (id < n);
    if (in_bounds) {
        value = src[id];
        digit = (value >> shift) & (RADIX_SIZE - 1);
    }

    for (uint d = 0u; d < RADIX_SIZE; ++d) {
        digit_prefix_counts[local_id][d] = (in_bounds && digit == d) ? 1u : 0u;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1u; stride < GROUP_SIZE; stride <<= 1) {
        const bool has_previous = (local_id >= stride);
        uint previous[RADIX_SIZE];
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            previous[d] = has_previous ? digit_prefix_counts[local_id - stride][d] : 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            digit_prefix_counts[local_id][d] += previous[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (id < n) {
        const uint dst_index = offsets[group_id * RADIX_SIZE + digit]
                             + (digit_prefix_counts[local_id][digit] - 1u);
        rassert(dst_index < n, 523451232);
        dst[dst_index] = value;
    }
}
