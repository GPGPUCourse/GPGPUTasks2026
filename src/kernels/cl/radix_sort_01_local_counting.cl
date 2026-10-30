#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    uint cur_bit)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);

    __local uint cnt0[GROUP_SIZE];
    cnt0[local_index] = !((buffer1[index] >> cur_bit) & 1);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int level = 1; level < GROUP_SIZE; level += level) {
        if (local_index & level) {
            cnt0[local_index] += cnt0[(local_index ^ level) | (level - 1)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        buffer2[index / GROUP_SIZE] = cnt0[GROUP_SIZE - 1];
    }
}
