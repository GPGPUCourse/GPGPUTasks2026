#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* group_counts,
    __global       uint* group_offsets,
    unsigned int group_count)
{
    uint total_zeros = 0;
    for (uint group = 0; group < group_count; ++group) {
        total_zeros += group_counts[group * 2];
    }

    uint zero_prefix = 0;
    uint one_prefix = 0;
    for (uint group = 0; group < group_count; ++group) {
        const uint zeros = group_counts[group * 2];
        const uint ones = group_counts[group * 2 + 1];
        group_offsets[group * 2] = zero_prefix;
        group_offsets[group * 2 + 1] = total_zeros + one_prefix;
        zero_prefix += zeros;
        one_prefix += ones;
    }

    group_offsets[group_count * 2] = total_zeros;
}
