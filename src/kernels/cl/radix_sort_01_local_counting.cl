#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const size_t i = get_global_id(0);
    const uint lid = get_local_id(0);
    __local uint counts[GROUP_SIZE];

    if (i < a1) {
        counts[lid] = ((buffer1[i] >> a2) & 1) == 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const uint value = lid >= offset ? counts[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        counts[lid] += value;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < a1) {
        buffer2[i] = counts[lid];
    }
}
