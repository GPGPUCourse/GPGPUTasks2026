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
     __global uint* pref_sum,
     __global uint* block,
     unsigned int n,
     unsigned int bit)
{
    uint g = get_global_id(0);
    uint l = get_local_id(0);
    uint gr = get_group_id(0);
    __local uint buf[GROUP_SIZE];
    buf[l] = g < n ? 1 - ((input[g] >> bit) & 1) : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = 2; step <= GROUP_SIZE; step <<= 1) {
        uint i = l + 1;
        if (i <= GROUP_SIZE / step) {
            buf[i * step - 1] += buf[i * step - 1 - step / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint step = GROUP_SIZE / 2; step > 1; step >>= 1) {
        uint i = l + 1;
        if (i < GROUP_SIZE / step) {
            buf[i * step + step / 2 - 1] += buf[i * step - 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (g < n) {
        pref_sum[g] = buf[l];
    }
    if (l == GROUP_SIZE - 1) {
        block[gr] = buf[l];
    }
}
