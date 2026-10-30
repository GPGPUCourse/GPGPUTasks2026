#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    uint n)
{
    uint index = get_global_id(0);
    uint local_index = get_local_id(0);

    __local uint helper[GROUP_SIZE];

    helper[local_index] = (index < n ? buffer1[index] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int level = 1; level < GROUP_SIZE; level += level) {
        if (local_index & level) {
            helper[local_index] += helper[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        buffer2[index / GROUP_SIZE] = helper[GROUP_SIZE - 1];
    }
}
