#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* group_counts,
    __global       uint* next_group_counts,
    unsigned int group_count)
{
    const uint global_id = get_global_id(0);

    const uint pair_id = global_id / RADIX_SIZE;
    const uint bucket = global_id % RADIX_SIZE;

    const uint left_group = 2 * pair_id;
    const uint right_group = left_group + 1;

    if (left_group >= group_count) {
        return;
    }

    const uint left_index = left_group * RADIX_SIZE + bucket;
    const uint output_index = pair_id * RADIX_SIZE + bucket;

    uint sum = group_counts[left_index];

    if (right_group < group_count) {
        const uint right_index = left_index + RADIX_SIZE;
        sum += group_counts[right_index];
    }

    next_group_counts[output_index] = sum;
}
