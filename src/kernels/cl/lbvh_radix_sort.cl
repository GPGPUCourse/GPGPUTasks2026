#include "../defines.h"

__kernel void lbvh_radix_scan(__global const uint* values,
                              __global uint* prefixes,
                              __global uint* block_sums,
                              uint n, 
                              uint bit, 
                              uint count_bits)
{
    __local uint temp[2 * GROUP_SIZE];
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t block_size = 2 * group_size;
    size_t block_start = get_group_id(0) * block_size;

    size_t x1 = block_start + local_id;
    size_t x2 = x1 + group_size;
    temp[local_id] = x1 < n ? (count_bits ? ((values[x1] >> bit) & 1u) == 0 : values[x1]) : 0;
    temp[local_id + group_size] = x2 < n ? (count_bits ? ((values[x2] >> bit) & 1u) == 0 : values[x2]) : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint cur = 1; cur < block_size; cur <<= 1) {
        size_t right = (local_id + 1) * (2 * cur) - 1;
        if (right < block_size) {
            temp[right] += temp[right - cur];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        block_sums[get_group_id(0)] = temp[block_size - 1];
        temp[block_size - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint cur = block_size / 2; cur > 0; cur >>= 1) {
        size_t right = (local_id + 1) * (2 * cur) - 1;
        if (right < block_size) {
            uint left_sum = temp[right - cur];
            uint prefix = temp[right];
            temp[right - cur] = prefix;
            temp[right] = prefix + left_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x1 < n) {
        prefixes[x1] = temp[local_id];
    }
    if (x2 < n) {
        prefixes[x2] = temp[local_id + group_size];
    }
}

__kernel void lbvh_radix_accumulate(__global const uint* block_prefixes,
                                    __global uint* prefixes, uint n)
{
    size_t i = get_global_id(0);
    if (i < n) {
        prefixes[i] += block_prefixes[i / (2 * GROUP_SIZE)];
    }
}

__kernel void lbvh_radix_scatter(__global const uint* codes,
                                 __global const uint* indices,
                                 __global const uint* local_prefixes,
                                 __global const uint* block_prefixes,
                                 __global const uint* total_zeros,
                                 __global uint* output_codes,
                                 __global uint* output_indices, uint n, uint bit)
{
    size_t i = get_global_id(0);
    if (i < n) {
        uint block = i / (2 * GROUP_SIZE);
        uint zeros_before = local_prefixes[i] + (block == 0 ? 0 : block_prefixes[block]);
        uint value = codes[i];
        size_t destination = ((value >> bit) & 1u) == 0
            ? zeros_before : total_zeros[0] + i - zeros_before;
        output_codes[destination] = value;
        output_indices[destination] = indices[i];
    }
}
