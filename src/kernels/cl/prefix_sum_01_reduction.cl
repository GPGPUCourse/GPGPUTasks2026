#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n,
    unsigned int input_offset,
    unsigned int output_offset)
{
    size_t i = get_global_id(0);
    if (i >= n / 2) {
        return;
    }

    next_pow2_sum[output_offset + i] = pow2_sum[input_offset + 2 * i] + pow2_sum[input_offset + 2 * i + 1];
}
