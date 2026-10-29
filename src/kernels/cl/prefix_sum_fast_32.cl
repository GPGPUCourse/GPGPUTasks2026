#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void prefix_sum_fast_in_group_32(
    __global const uint* cur_arr,
    __global       uint* new_arr,
    __global       uint* helper_arr
)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    __local uint arr[128];

    arr[local_index] = cur_arr[index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int block = 1; block <= 128; block += block) {
        if (local_index & block) {
            arr[local_index] += arr[(local_index ^ block) | (block - 1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    new_arr[index] = arr[local_index];
    if (local_index == 0) {
        helper_arr[index / 128] = arr[127];
    }
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void prefix_sum_fast_32(
    __global const uint* cur_arr,
    __global const uint* cur_helper_arr,
    __global       uint* new_arr,
    __global       uint* new_helper_arr,
    int n,
    uint block_size,
    uint helper_size
) {
    uint index = get_global_id(0);
    uint local_index = get_local_id(0);

    __local uint helper[128];
    uint my_index = (index / block_size);
    uint start = my_index & (~127);

    uint count = my_index - start;

    helper[local_index] = (start + local_index < helper_size && local_index < count ? cur_helper_arr[start + local_index] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int block = 1; block <= count; block += block) {
        if (local_index & block) {
            helper[local_index] += helper[(local_index ^ block) | (block - 1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint sum_before = (my_index > start ? helper[my_index - start - 1] : 0);


    if (local_index == 0 && index + 127 < n && ((index + 127) & (block_size * 128 - 1)) == block_size * 128 - 1) {
        new_helper_arr[my_index / 128] = sum_before + cur_arr[index + 127];
    }

    new_arr[index] = cur_arr[index] + sum_before;
}
