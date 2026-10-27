#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* offsets,
    __global const uint* totals,
    unsigned int n)
{
    if (get_global_id(0) != 0)
        return;

    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    uint global_base = 0u;

    for (uint digit = 0; digit < RADIX_SIZE; ++digit) {
        uint block_prefix = 0u;

        for (uint group = 0; group < num_groups; ++group) {
            const uint index = digit * num_groups + group;
            const uint count = offsets[index];

            offsets[index] = global_base + block_prefix;
            block_prefix += count;
        }

        global_base += totals[digit];
    }
}
