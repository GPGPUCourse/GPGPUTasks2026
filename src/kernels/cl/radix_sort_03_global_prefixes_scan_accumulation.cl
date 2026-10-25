#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* inclusive_prefixes,
    __global       uint* offsets,
    uint n)
{
    uint index = get_global_id(0);
    if (index < n) {
        offsets[index] = index == 0 ? 0 : inclusive_prefixes[index - 1];
    }
}
