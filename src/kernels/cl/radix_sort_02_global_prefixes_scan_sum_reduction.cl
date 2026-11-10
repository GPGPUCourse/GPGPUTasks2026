#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global uint* prefixes,
    __global uint* sums,
    unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    uint value = gid < n ? prefixes[gid] : 0;
    __local uint scan[GROUP_SIZE];

    scan[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = 1; step < GROUP_SIZE; step *= 2) {
        uint add = lid >= step ? scan[lid - step] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        prefixes[gid] = scan[lid] - value;
    }
    if (lid == GROUP_SIZE - 1) {
        sums[get_group_id(0)] = scan[lid];
    }
}
