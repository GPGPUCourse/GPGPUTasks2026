#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,
    __global const uint* pow2_sums,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }

    uint prefix_length = (uint) i + 1;
    uint sum = (prefix_length & 1) ? input[i] : 0;
    uint level_size = n / 2;
    size_t level_offset = 0;
    for (uint block = prefix_length >> 1; block != 0; block >>= 1) {
        if (block & 1) {
            sum += pow2_sums[level_offset + block - 1];
        }
        level_offset += level_size;
        level_size /= 2;
    }
    prefix_sum_accum[i] = sum;
}
