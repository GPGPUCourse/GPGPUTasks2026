#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* offsets,
    __global       uint* a,
    unsigned int n,
    unsigned int level)
{
    const unsigned int k = get_global_id(0);
    const unsigned int i = k >> (8 * level);

    if (k < n && i % GROUP_SIZE != 0)
        a[k] += offsets[i];
}
