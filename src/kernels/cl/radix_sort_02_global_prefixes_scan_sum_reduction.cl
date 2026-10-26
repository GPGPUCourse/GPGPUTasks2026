#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global       uint* block_prefix,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint values[RADIX_BLOCK_SIZE];
    const uint lid = get_local_id(0);
    const size_t first = get_group_id(0) * RADIX_BLOCK_SIZE + lid;
    const size_t second = first + GROUP_SIZE;
    const uint a = first < n ? input[first] : 0;
    const uint b = second < n ? input[second] : 0;
    values[lid] = a;
    values[lid + GROUP_SIZE] = b;

    // Blelloch up-sweep: build a tree of partial sums.
    for (uint stride = 1; stride < RADIX_BLOCK_SIZE; stride *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < RADIX_BLOCK_SIZE)
            values[right] += values[right - stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        block_sums[get_group_id(0)] = values[RADIX_BLOCK_SIZE - 1];
        values[RADIX_BLOCK_SIZE - 1] = 0;
    }

    // Down-sweep: propagate the sum preceding each subtree.
    for (uint stride = RADIX_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < RADIX_BLOCK_SIZE) {
            const uint left_sum = values[right - stride];
            values[right - stride] = values[right];
            values[right] += left_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // The tree produced an exclusive scan; add the original values for inclusive scan.
    if (first < n)
        block_prefix[first] = values[lid] + a;
    if (second < n)
        block_prefix[second] = values[lid + GROUP_SIZE] + b;
}
