#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* prefix_sum,
    __global const uint* prev_level,
    unsigned int n)
{
    const unsigned int ind = get_global_id(0);
    if (ind >= n) {
        return;
    }

    if (ind % 2) {
        prefix_sum[ind] = prev_level[ind / 2];
    } else if (ind / 2 > 0) {
        prefix_sum[ind] += prev_level[ind / 2 - 1];
    }
}
