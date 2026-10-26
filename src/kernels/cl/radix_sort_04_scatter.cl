#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* counts_prefix_sum,
    __global       uint* output,
    unsigned int n,
    unsigned int shift)
{
    __local uint digits[GROUP_SIZE];
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);

    const uint value = i < n ? input[i] : 0;
    const uint digit = i < n ? (value >> shift) & (RADIX_BINS - 1) : RADIX_BINS;
    digits[lid] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i >= n) {
        return;
    }

    uint rank = 0;
    for (uint k = 0; k < lid; ++k) {
        rank += digits[k] == digit;
    }

    const uint index = digit * get_num_groups(0) + get_group_id(0);
    const uint offset = index == 0 ? 0 : counts_prefix_sum[index - 1];
    output[offset + rank] = value;
}
