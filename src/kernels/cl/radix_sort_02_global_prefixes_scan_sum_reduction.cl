#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"
#include "helpers/exclusive_scan.cl"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global uint* prefixes,
    __global uint* block_sums,
    uint n)
{
    size_t index = get_global_id(0);
    uint lane = get_local_id(0);
    __local uint sums[GROUP_SIZE];
    sums[lane] = index < n ? prefixes[index] : 0;
    uint total = exclusive_scan(sums, lane);
    if (index < n) prefixes[index] = sums[lane];
    if (lane == 0) block_sums[get_group_id(0)] = total;
}
