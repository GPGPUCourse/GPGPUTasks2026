#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const uint i = get_global_id(0);
    if (i >= (n + 1) / 2)
        return;
    uint left = pow2_sum[i * 2 + 0];
    uint right = (i * 2 + 1 < n) ? pow2_sum[i * 2 + 1] : 0;
    next_pow2_sum[i] = left + right;
}
