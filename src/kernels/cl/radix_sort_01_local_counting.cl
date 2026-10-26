#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* src,
    __global       uint* counts,
    uint n,
    uint shift)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint local_counts[RADIX_SIZE];
    if (local_id < RADIX_SIZE) {
        local_counts[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < n) {
        const uint value = src[id];
        const uint digit = (value >> shift) & (RADIX_SIZE - 1);
        atomic_inc(&local_counts[digit]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX_SIZE) {
        counts[group_id * RADIX_SIZE + local_id] = local_counts[local_id];
    }
}
