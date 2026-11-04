#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* group_prefixes,
    __global uint* prefixes,
    unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid < n) {
        prefixes[gid] += group_prefixes[gid / GROUP_SIZE];
    }
}
