#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* counts,
    uint n,
    uint shift)
{
    uint lane = get_local_id(0);
    uint index = get_global_id(0);
    uint groups = get_num_groups(0);
    __local uint histogram[RADIX_BUCKETS];
    if (lane < RADIX_BUCKETS) {
        histogram[lane] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n) {
        atomic_inc(&histogram[(values[index] >> shift) & (RADIX_BUCKETS - 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < RADIX_BUCKETS) {
        counts[lane * groups + get_group_id(0)] = histogram[lane];
    }
}
