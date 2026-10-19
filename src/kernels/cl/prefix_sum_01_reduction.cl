#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* buffer_pow2_sum,
    unsigned int input_begin,
    unsigned int output_begin)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int n = output_begin - input_begin;

    __local uint local_pow2_sum[GROUP_SIZE];
    local_pow2_sum[local_index] = index < n ? buffer_pow2_sum[input_begin + index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < GROUP_SIZE / 2) {
        const unsigned int output_index = output_begin + get_group_id(0) * (GROUP_SIZE / 2) + local_index;
        buffer_pow2_sum[output_index] = local_pow2_sum[local_index * 2] + local_pow2_sum[local_index * 2 + 1];
    }
}
