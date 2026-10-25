#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* group_counts,
    __global uint* group_prefixes,
    uint groups_count)
{
    const uint index = get_global_id(0);
    if (index < groups_count) {
        group_prefixes[index] = group_counts[index]
            + (index > 0 ? group_counts[index - 1] : 0u);
    }
}
