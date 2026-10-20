#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global uint* prefixes,
    __global uint* sums,
    uint n,
    uint bit)
{
    // TODO
    uint i = get_global_id(0), lid = get_local_id(0);
    __local uint counts[GROUP_SIZE];
    counts[lid] = i < n ? ((input[i] >> bit) & 1) == 0 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        uint add = lid >= offset ? counts[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        counts[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n) {
        prefixes[i] = counts[lid];
    }
    if (lid == GROUP_SIZE - 1) {
        sums[get_group_id(0)] = counts[lid];
    }
}
