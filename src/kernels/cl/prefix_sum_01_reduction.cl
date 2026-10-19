#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* prefix_sum_accum,
    __global const uint* scanned_group_sums,
    unsigned int n)
{
    const uint id = get_global_id(0);
    const uint group_id = get_group_id(0);

    const uint block_id = id / (GROUP_SIZE * 2);

    if (id < n && block_id > 0) {
        prefix_sum_accum[id] += scanned_group_sums[block_id - 1];
    }
}
