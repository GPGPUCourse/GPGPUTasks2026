#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* block_prefixes,
    __global       uint* output,
    unsigned int n,
    unsigned int bit)
{
    __local uint zeros[RADIX_BLOCK_SIZE];
    const uint lid = get_local_id(0);
    const size_t block = get_group_id(0);
    const size_t first = block * RADIX_BLOCK_SIZE + lid;
    const size_t second = first + GROUP_SIZE;
    const uint first_key = first < n ? input[first] : 0;
    const uint second_key = second < n ? input[second] : 0;
    const uint a = first < n && ((first_key >> bit) & 1u) == 0;
    const uint b = second < n && ((second_key >> bit) & 1u) == 0;
    zeros[lid] = a;
    zeros[lid + GROUP_SIZE] = b;

    // Exclusive scan of zero flags gives each element its stable local rank.
    for (uint stride = 1; stride < RADIX_BLOCK_SIZE; stride *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < RADIX_BLOCK_SIZE)
            zeros[right] += zeros[right - stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
        zeros[RADIX_BLOCK_SIZE - 1] = 0;
    for (uint stride = RADIX_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < RADIX_BLOCK_SIZE) {
            const uint left_sum = zeros[right - stride];
            zeros[right - stride] = zeros[right];
            zeros[right] += left_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint preceding_zeros = block > 0 ? block_prefixes[block - 1] : 0;
    const size_t blocks = ((size_t)n + RADIX_BLOCK_SIZE - 1) / RADIX_BLOCK_SIZE;
    const uint total_zeros = block_prefixes[blocks - 1];
    if (first < n) {
        const uint inclusive_zeros = preceding_zeros + zeros[lid] + a;
        const size_t destination = a ? inclusive_zeros - 1 : total_zeros + first - inclusive_zeros;
        output[destination] = first_key;
    }
    if (second < n) {
        const uint inclusive_zeros = preceding_zeros + zeros[lid + GROUP_SIZE] + b;
        const size_t destination = b ? inclusive_zeros - 1 : total_zeros + second - inclusive_zeros;
        output[destination] = second_key;
    }
}
