#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* histogram,
    __global uint* totals,
    unsigned int n)
{
    const uint digit = get_global_id(0);
    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    if (digit >= RADIX_SIZE)
        return;

    uint total = 0u;

    for (uint group = 0; group < num_groups; ++group) {
        total += histogram[digit * num_groups + group];
    }

    totals[digit] = total;
}
