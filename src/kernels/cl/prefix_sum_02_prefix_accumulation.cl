#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* scanned_block_sums,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    const unsigned int group_id = get_group_id(0);
    if (group_id == 0)
        return;

    const uint block_offset = scanned_block_sums[group_id - 1];
    const unsigned int a_index = group_id * PREFIX_SUM_BLOCK_SIZE + get_local_id(0);
    const unsigned int b_index = a_index + GROUP_SIZE;
    if (a_index < n)
        prefix_sum_accum[a_index] += block_offset;
    if (b_index < n)
        prefix_sum_accum[b_index] += block_offset;
}
