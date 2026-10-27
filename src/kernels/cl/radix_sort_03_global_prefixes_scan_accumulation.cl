#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global uint* buffer2,
    unsigned int n,
    unsigned int unused)
{
    if (get_global_id(0) != 0)
        return;

    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    uint zeros_prefix = 0;
    for (uint g = 0; g < num_groups; ++g) {
        const uint count = buffer2[g];
        buffer2[g] = zeros_prefix;
        zeros_prefix += count;
    }

    uint ones_prefix = 0;
    for (uint g = 0; g < num_groups; ++g) {
        const uint index = num_groups + g;
        const uint count = buffer2[index];
        buffer2[index] = ones_prefix;
        ones_prefix += count;
    }
    buffer2[2 * num_groups] = zeros_prefix;
}
