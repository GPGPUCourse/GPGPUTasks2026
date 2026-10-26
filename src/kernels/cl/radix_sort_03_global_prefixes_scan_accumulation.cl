#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* values,
    __global const uint* group_offsets,
    __global       uint* positions,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    uint value_bit;
    if (gid >= n) {
        value_bit = 0;
    } else {
        value_bit = (values[gid] >> bit) & 1u;
    }

    __local uint zero_prefix[GROUP_SIZE];
    const uint local_id = get_local_id(0);
    const uint group = get_group_id(0);
    zero_prefix[local_id] = value_bit == 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        uint previous = 0;
        if (local_id >= offset) {
            previous = zero_prefix[local_id - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        zero_prefix[local_id] += previous;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        const uint zeros_before = zero_prefix[local_id] - (value_bit == 0);
        const uint rank = value_bit == 0 ? zeros_before : local_id - zero_prefix[local_id];
        positions[gid] = group_offsets[group * 2 + value_bit] + rank;
    }
}
