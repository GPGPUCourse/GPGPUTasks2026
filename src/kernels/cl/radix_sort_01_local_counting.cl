#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* a,
    __global       uint* histogram,
    unsigned int n,
    unsigned int bit)
{
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint count[RADIX];

    if (local_id < RADIX) {
        count[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint g1 = group_id * 2 * GROUP_SIZE + local_id;
    uint g2 = g1 + GROUP_SIZE;

    if (g1 < n) {
        const uint d = (a[g1] >> bit) & (RADIX - 1);
        atomic_inc((volatile __local uint*)&count[d]);
    }
    if (g2 < n) {
        const uint d = (a[g2] >> bit) & (RADIX - 1);
        atomic_inc((volatile __local uint*)&count[d]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX) {
        histogram[group_id * RADIX + local_id] = count[local_id];
    }
}
