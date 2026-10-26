#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* locally_sorted,
    __global const uint* prefixes,
    __global uint* output, uint n, uint shift, uint groups)
{
    size_t index = get_global_id(0);
    uint lane = get_local_id(0);
    uint value = index < n ? locally_sorted[index] : 0;
    uint digit = (value >> shift) & (RADIX_BUCKETS - 1);
    __local uint digits[GROUP_SIZE];
    __local uint starts[RADIX_BUCKETS];
    digits[lane] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n && (lane == 0 || digit != digits[lane - 1]))
        starts[digit] = lane;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n) {
        uint destination = prefixes[digit * groups + get_group_id(0)] + lane - starts[digit];
        output[destination] = value;
    }
}
