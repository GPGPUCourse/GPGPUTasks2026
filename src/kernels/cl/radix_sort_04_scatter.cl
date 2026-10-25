#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* local_sorted,
    __global const uint* prefixes,
    __global uint* output,
    unsigned int n,
    unsigned int shift)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);

    __local uint digits[GROUP_SIZE];
    __local uint starts[16];

    const uint value = global_id < n
        ? local_sorted[global_id]
        : 0xffffffffu;

    const uint digit = (value >> shift) & 15u;

    digits[local_id] = digit;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0 || digit != digits[local_id - 1]) {
        starts[digit] = local_id;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        const uint offset = prefixes[digit * get_num_groups(0) + get_group_id(0)];
        output[offset + local_id - starts[digit]] = value;
    }
}
