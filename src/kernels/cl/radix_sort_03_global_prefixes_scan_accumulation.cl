#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sums,
    __global       uint* group_prefixes,
    unsigned int group_count,
    unsigned int pow2)
{
    const uint global_id = get_global_id(0);

    const uint group_id = global_id / RADIX_SIZE;
    const uint bucket = global_id % RADIX_SIZE;

    if (group_id >= group_count) {
        return;
    }

    if (((group_id + 1) >> pow2) & 1u) {
        const uint sum_id = ((group_id + 1) >> pow2) - 1;
        group_prefixes[group_id * RADIX_SIZE + bucket] += pow2_sums[sum_id * RADIX_SIZE + bucket];
    }
}
