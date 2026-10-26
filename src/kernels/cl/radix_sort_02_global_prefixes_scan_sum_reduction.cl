#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global       uint* output,
    __global       uint* accumulated,
    uint bit,
    uint is_bit_map_step,
    uint n)
{
    const uint item_idx = get_global_id(0);
    const uint group_idx = get_local_id(0);

    __local uint buf[GROUP_SIZE];
    if (item_idx >= n) {
        buf[group_idx] = 0;
    } else {
        buf[group_idx] = is_bit_map_step == 1 ? ((input[item_idx] >> bit) & 1) == 0 : input[item_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint level = 1; level < GROUP_SIZE; level *= 2) {
        const uint delta = level <= group_idx ? buf[group_idx - level] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        buf[group_idx] += delta;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (item_idx < n) {
        output[item_idx] = buf[group_idx];
    }
    if (group_idx == 0) {
        accumulated[get_group_id(0)] = buf[GROUP_SIZE - 1];
    }
}
