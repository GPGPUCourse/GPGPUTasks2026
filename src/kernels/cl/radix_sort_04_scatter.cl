#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* prefix,
    __global const uint* input,
    __global       uint* output,
    unsigned int n,
    unsigned int bit)
{
    // TODO
    const uint global_id = get_global_id(0);
    if (global_id >= n) return;

    int count_zero = prefix[n - 1] + (((input[n-1] >> bit) & 1u) == 0u);

    int cur_bit = ((input[global_id] >> bit) & 1u);
    int index_out = cur_bit ? count_zero + (global_id - prefix[global_id]) : prefix[global_id];
    output[index_out] = input[global_id];
}