#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* histogram,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t groups = get_num_groups(0);
    __local uint counts[RADIX_BUCKETS];
    if (lid < RADIX_BUCKETS) {
        counts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const size_t index = group * RADIX_BLOCK_SIZE + k * GROUP_SIZE + lid;
        if (index < n) {
            atomic_inc(&counts[(input[index] >> shift) & (RADIX_BUCKETS - 1)]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS) {
        histogram[lid * groups + group] = counts[lid];
    }
}
