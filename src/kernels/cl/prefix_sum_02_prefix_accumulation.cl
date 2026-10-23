#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* blocks,
    __global       uint* output,
    unsigned int n)
{
    const uint global_index = get_global_id(0);
    const uint block = get_group_id(0);

    if (global_index < n && block > 0) {
        output[global_index] += blocks[block - 1];
    }
}
