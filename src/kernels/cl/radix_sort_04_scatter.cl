#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* prefixes,
    __global       uint* output,
    unsigned int n,
    unsigned int group_count,
    unsigned int bit_offset)
{
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    const uint base = group_id * GROUP_SIZE;

    const uint valid_count = min((uint)GROUP_SIZE, n - min(n, base));

    __local uint data[GROUP_SIZE];
    __local uint temp[GROUP_SIZE];
    __local uint scan[GROUP_SIZE];

    __local uint local_bucket_starts[RADIX_SIZE];
    __local uint global_bucket_starts[RADIX_SIZE];

    if (local_id < valid_count) {
        data[local_id] = values[base + local_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint bit = 0; bit < RADIX_BITS; ++bit) {
        uint is_zero = 0;

        if (local_id < valid_count) {
            const uint value = data[local_id];
            is_zero = (((value >> (bit_offset + bit)) & 1u) == 0);
        }

        scan[local_id] = is_zero;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            uint add = 0;

            if (local_id >= offset) {
                add = scan[local_id - offset];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_id >= offset) {
                scan[local_id] += add;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id < valid_count) {
            const uint total_zeros = scan[valid_count - 1];
            const uint zeros_before = scan[local_id] - is_zero;

            uint new_index;

            if (is_zero) {
                new_index = zeros_before;
            } else {
                const uint ones_before = local_id - zeros_before;
                new_index = total_zeros + ones_before;
            }

            temp[new_index] = data[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < valid_count) {
            data[local_id] = temp[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        uint local_start = 0;
        uint global_start = 0;

        for (uint bucket = 0; bucket < RADIX_SIZE; ++bucket) {
            local_bucket_starts[bucket] = local_start;
            global_bucket_starts[bucket] = global_start;

            const uint current_prefix = prefixes[group_id * RADIX_SIZE + bucket];
            const uint previous_prefix = group_id == 0 ? 0 : prefixes[(group_id - 1) * RADIX_SIZE + bucket];
            const uint local_count = current_prefix - previous_prefix;

            local_start += local_count;

            const uint total_count = prefixes[(group_count - 1) * RADIX_SIZE + bucket];
            global_start += total_count;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id >= valid_count) {
        return;
    }

    const uint value = data[local_id];
    const uint bucket = (value >> bit_offset) & (RADIX_SIZE - 1);
    const uint local_offset = local_id - local_bucket_starts[bucket];
    const uint group_offset = group_id == 0 ? 0 : prefixes[(group_id - 1) * RADIX_SIZE + bucket];

    const uint global_index = global_bucket_starts[bucket] + group_offset + local_offset;

    output[global_index] = value;
}