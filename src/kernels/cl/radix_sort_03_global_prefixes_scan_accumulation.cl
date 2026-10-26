#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sums,
    __global uint* global_one_prefix,
    uint num_groups,
    uint pow2)
{
    const uint g = get_global_id(0);

    if (g >= num_groups) {
        return;
    }

    const uint block_size = 1u << pow2;

    if ((g & block_size) != 0u) {
        const uint block_index = (g / block_size) - 1u;
        global_one_prefix[g] += pow2_sums[block_index];
    }
}