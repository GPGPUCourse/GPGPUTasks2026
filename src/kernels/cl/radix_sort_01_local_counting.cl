#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int n,
    unsigned int bit)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint count[RADIX];

    if (local_id < RADIX) {
        count[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < n) {
        const uint d = (buffer1[id] >> bit) & (RADIX - 1);
        atomic_inc((volatile __local uint*)&count[d]);
    }

    if (local_id < RADIX) {
        buffer2[group_id * GROUP_SIZE + local_id] = count[loacl_id];
    }
}
