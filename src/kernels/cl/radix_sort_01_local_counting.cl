#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* zero_counts,
    unsigned int n,
    unsigned int bit)
{
    __local uint counts[GROUP_SIZE];
    const uint lid = get_local_id(0);
    const size_t first = get_group_id(0) * RADIX_BLOCK_SIZE + lid;
    const size_t second = first + GROUP_SIZE;
    counts[lid] = (first < n && ((input[first] >> bit) & 1u) == 0)
                + (second < n && ((input[second] >> bit) & 1u) == 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride)
            counts[lid] += counts[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        zero_counts[get_group_id(0)] = counts[0];
}
