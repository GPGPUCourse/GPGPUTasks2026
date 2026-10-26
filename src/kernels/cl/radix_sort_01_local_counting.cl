#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* group_counts,
    unsigned int n,
    unsigned int bit)
{
    __local uint counts[2];
    const uint local_id = get_local_id(0);
    const uint index = get_global_id(0);

    if (local_id < 2) {
        counts[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (index < n) {
        atomic_inc(&counts[(values[index] >> bit) & 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 2) {
        group_counts[get_group_id(0) * 2 + local_id] = counts[local_id];
    }
}
