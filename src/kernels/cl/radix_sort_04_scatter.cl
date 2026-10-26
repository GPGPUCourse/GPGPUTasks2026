#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Digit-major scan combines all previous buckets and previous workgroups.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* local_digit_rank,
    __global const uint* global_prefix,
    __global uint* output,
    uint n,
    uint groups,
    uint shift)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint value = input[i];
    const uint digit = (value >> shift) & 15u;
    const uint g = get_group_id(0);
    const uint dst = global_prefix[digit * groups + g] + local_digit_rank[i];
    output[dst] = value;
}
