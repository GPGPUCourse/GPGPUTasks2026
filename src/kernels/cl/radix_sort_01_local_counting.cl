#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input_buffer,
    __global uint* local_zero_prefixes,
    __global uint* group_zero_counts,
    uint n,
    uint bit)
{
    const uint global_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);

    __local uint zero_prefixes[GROUP_SIZE];
    zero_prefixes[local_index] = global_index < n
        ? 1u - ((input_buffer[global_index] >> bit) & 1u)
        : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        const uint value = local_index >= offset ? zero_prefixes[local_index - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        zero_prefixes[local_index] += value;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_index < n) {
        local_zero_prefixes[global_index] = zero_prefixes[local_index];
    }
    if (local_index == GROUP_SIZE - 1) {
        group_zero_counts[group_index] = zero_prefixes[local_index];
    }
}
