#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    __global const uint* parent_prefix)
{
    // TODO
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }

    prefix_sum_accum[i] = (i % 2) ? parent_prefix[i / 2]
        : pow2_sum[i] + (i > 0 ? parent_prefix[i / 2 - 1] : 0);
}
