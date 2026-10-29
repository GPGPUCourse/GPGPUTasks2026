#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int prev_n,
    unsigned int new_n)
{
    int index = get_global_id(0);
    int local_index = get_local_id(0);
    __local uint buffer[GROUP_SIZE * 2];

    int start_index = 2 * (index - local_index);

    buffer[local_index] = (start_index + local_index < prev_n ? pow2_sum[start_index + local_index] : 0);
    buffer[local_index + GROUP_SIZE] = (start_index + local_index + GROUP_SIZE < prev_n ? pow2_sum[start_index + local_index + GROUP_SIZE] : 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index < new_n) {
        next_pow2_sum[index] = buffer[2 * local_index] + buffer[2 * local_index + 1];
    }
}
