#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,
    __global const uint* buffer_pow2_sum,
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int min_precomputed_pow2,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local uint local_input[GROUP_SIZE];
    local_input[local_index] = index < n ? input[index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index >= n) return;

    const unsigned int window_size = 1 << min_precomputed_pow2;

    unsigned int sum = 0;

    rassert(index % window_size == local_index % window_size, 230910185);
    for (unsigned int i = 0; i < (local_index + 1) % window_size; ++i) {
        sum += local_input[local_index + 1 + i - (local_index + 1) % window_size];
    }

    for (unsigned pow2 = min_precomputed_pow2, offset = 0; ((index + 1) >> pow2) > 0; ++pow2) {
        if (((index + 1) >> pow2) & 1) {
            unsigned int input_index = ((index + 1) / (1 << pow2) - 1);
            sum += buffer_pow2_sum[offset + input_index];
        }

        offset += n >> pow2;
    }

    prefix_sum_accum[index] = sum;
}
