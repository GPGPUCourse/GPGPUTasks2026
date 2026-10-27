#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* values,
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global uint* buffer3,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    if (gid >= n)
        return;

    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint group = gid / GROUP_SIZE;

    const uint encoded = buffer1[gid];
    const uint digit = encoded & 1u;
    const uint rank = (encoded >> 1) & 255u;

    uint dst;
    if (digit == 0u) {
        dst = buffer2[group] + rank;
    } else {
        dst = buffer2[2 * num_groups]
            + buffer2[num_groups + group]
            + rank;
    }
    buffer3[dst] = values[gid];
}