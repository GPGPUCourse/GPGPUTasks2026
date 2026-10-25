#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* block_prefixes,
    __global uint* prefixes,
    unsigned int n)
{
    const uint global_id = get_global_id(0);
    if (global_id < n) {
        prefixes[global_id] += block_prefixes[get_group_id(0)];
    }
}
