#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* scanned_block_sums,
    __global       uint* prefix_sum,
    unsigned int n)
{
    const size_t i = get_global_id(0);
    if (i >= n)
        return;

    const size_t block = i / RADIX_BLOCK_SIZE;
    if (block > 0)
        prefix_sum[i] += scanned_block_sums[block - 1];
}
