#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* prefixes,
    __global const uint* block_prefixes,
    __global       uint* output,
    unsigned int n,
    unsigned int bit)
{
    const uint i = get_global_id(0);

    if (i >= n) {
        return;
    }

    const uint value = input[i];
    const uint group = get_group_id(0);
    const uint prefix = prefixes[i] + (group > 0 ? block_prefixes[group - 1] : 0);
    const uint total_zeros = block_prefixes[get_num_groups(0) - 1];

    const uint position = ((value >> bit) & 1) == 0
        ? prefix - 1
        : total_zeros + i - prefix;

    output[position] = value;
}
