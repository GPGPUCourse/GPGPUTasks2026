#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global       uint* prefix,
    __global       uint* group_sums,
    unsigned int n,
    int bit)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    int value = 0;
    if (bit == -1)
        value = global_id < n ? input[global_id] : 0;
    else
        value = global_id < n ? (((input[global_id] >> bit) & 1u) == 0u) : 0;

    __local uint cur_values[GROUP_SIZE];

    cur_values[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    int step = 1;
    for (step = 1; step < GROUP_SIZE; step <<= 1){
        if ((local_id + 1) % (step * 2) == 0)
            cur_values[local_id] += cur_values[local_id - step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0){
        group_sums[get_group_id(0)] = cur_values[GROUP_SIZE - 1];
        cur_values[GROUP_SIZE - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = GROUP_SIZE >> 1; i >= 1; i >>= 1){
        if ((local_id + 1) % (i * 2) == 0){
            int left = local_id - i;
            int cur_left = cur_values[left];
            cur_values[left] = cur_values[local_id];
            cur_values[local_id] += cur_left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_id < n)
        prefix[global_id] = cur_values[local_id];


}
