#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global       uint* prefix,
    __global const uint* group_prefixes,
    unsigned int n)
{
    const uint index = get_global_id(0);
    if (index >= n)
        return;

    const uint group = get_group_id(0);
    if (group > 0)
        prefix[index] += group_prefixes[group - 1];
}
