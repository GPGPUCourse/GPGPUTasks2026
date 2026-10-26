#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1)
{
    // TODO
    const size_t i = get_global_id(0);
    const uint lid = get_local_id(0);
    const size_t n = (a1 + GROUP_SIZE - 1) / GROUP_SIZE;
    __local uint counts[GROUP_SIZE];

    counts[lid] = 0;
    if (i < n) {
        const size_t end = min((i + 1) * GROUP_SIZE, (size_t)a1);
        counts[lid] = buffer1[end - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const uint value = lid >= offset ? counts[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        counts[lid] += value;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n) {
        buffer2[i] = counts[lid];
    }
}
