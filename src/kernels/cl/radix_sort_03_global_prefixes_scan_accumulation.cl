#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* input_prefixes,
    __global uint* output_prefixes,
    uint groups_count,
    uint offset)
{
    const uint index = get_global_id(0);
    if (index < groups_count) {
        output_prefixes[index] = input_prefixes[index]
            + (index >= offset ? input_prefixes[index - offset] : 0u);
    }
}
