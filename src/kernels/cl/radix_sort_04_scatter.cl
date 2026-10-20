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
    __global const uint* sums,
    __global uint* output,
    uint n,
    uint bit)
{
    // TODO
    uint i = get_global_id(0), group = get_group_id(0);
    if (i >= n) {
        return;
    }
    uint value = input[i];
    uint zeros = prefixes[i] + (group > 0 ? sums[group - 1] : 0);
    uint total = sums[(n - 1) / GROUP_SIZE];
    uint pos = ((value >> bit) & 1) == 0 ? zeros - 1 : total + i - zeros;
    output[pos] = value;
}