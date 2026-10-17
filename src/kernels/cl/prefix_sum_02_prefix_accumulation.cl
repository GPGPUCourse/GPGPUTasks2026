#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    const size_t block = get_group_id(0) + 1;
    const uint offset = pow2_sum[block - 1];
    
    const size_t first = block * (2 * GROUP_SIZE) + get_local_id(0);
    const size_t second = first + GROUP_SIZE;

    if (first < n) {
        prefix_sum_accum[first] += offset;
    }
    if (second < n) {
        prefix_sum_accum[second] += offset;
    }
}
