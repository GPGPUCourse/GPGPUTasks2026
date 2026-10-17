#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,            // Исходный массив
    __global const uint* levels,           // Уровни 1+
    __global       uint* prefix_sum_accum, // prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    const unsigned int prefix_len = i + 1;

    prefix_sum_accum[i] = (prefix_len & 1) ? input[i] : 0;

    unsigned int level_size = n;
    unsigned int offset = 0;
    for (unsigned int k = 1; (prefix_len >> k) != 0; ++k) {
        level_size = (level_size + 1) / 2;

        const unsigned int blocks = prefix_len >> k;
        if (blocks & 1) {
            prefix_sum_accum[i] += levels[offset + blocks - 1];
        }

        offset += level_size;
    }
}
