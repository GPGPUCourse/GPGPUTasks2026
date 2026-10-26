#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* values,
    __global const uint* local_prefixes,
    __global const uint* block_prefixes,
    __global const uint* total_zeros,
    __global uint* output,
    unsigned int n,
    unsigned int bit)
{
    // TODO
    size_t index = get_global_id(0);
    if (index < n) {
        uint value = values[index];
        uint zeros_before = local_prefixes[index] + block_prefixes[index / (2 * GROUP_SIZE)];
        size_t destination = ((value >> bit) & 1u) == 0
            ? zeros_before
            : total_zeros[0] + index - zeros_before;
        output[destination] = value;
    }
}
