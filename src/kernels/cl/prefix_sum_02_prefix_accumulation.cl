#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    int j = get_global_id(0);
    if (j >= n)
        return;

    int count = j + 1;
    int digit = (count >> pow2) & 15u;
    if (digit == 0) {
        return;
    }

    int end_block = (count >> pow2) - 1;
    int add = 0;
    for (int d = 0; d < digit; d++) {
        add += pow2_sum[end_block - d];
    }
    prefix_sum_accum[j] += add;
}
