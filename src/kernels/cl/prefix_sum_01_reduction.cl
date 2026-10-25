#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* values,
    __global       uint* prefix_sum,
    unsigned int n)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }

    uint value = values[index];
    if (index > 0) {
        value += values[index - 1];
    }
    prefix_sum[index] = value;
}
