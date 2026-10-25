#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_buffer,
    __global const uint* local_zero_prefixes,
    __global const uint* group_zero_prefixes,
    __global uint* output_buffer,
    uint n,
    uint bit,
    uint groups_count)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }

    const uint group_index = get_group_id(0);
    const uint zeros_before_group = group_index > 0
        ? group_zero_prefixes[group_index - 1]
        : 0u;
    const uint local_zero_prefix = local_zero_prefixes[index];
    const uint total_zeros = group_zero_prefixes[groups_count - 1];
    const uint is_one = (input_buffer[index] >> bit) & 1u;

    const uint output_index = is_one
        ? total_zeros + index - zeros_before_group - local_zero_prefix
        : zeros_before_group + local_zero_prefix - 1u;
    output_buffer[output_index] = input_buffer[index];
}
