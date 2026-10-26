#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* positions,
    __global       uint* output,
    unsigned int n)
{
    const uint gid = get_global_id(0);
    if (gid < n) {
        output[positions[gid]] = values[gid];
    }
}
