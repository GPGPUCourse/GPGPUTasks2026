#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* counts,
    __global const uint* block_sums,
    __global       uint* offsets,
    uint m)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint slice_counts[GROUP_SIZE];
    __local uint thread_totals[GROUP_SIZE][RADIX_SIZE];
    __local uint thread_prefixes[GROUP_SIZE][RADIX_SIZE];
    __local uint chunk_totals[RADIX_SIZE][RADIX_SIZE];
    __local uint chunk_prefixes[RADIX_SIZE][RADIX_SIZE];
    __local uint digit_bases[RADIX_SIZE];

    const uint value = (id < m) ? counts[id] : 0u;
    slice_counts[local_id] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint acc_total[RADIX_SIZE];
    uint acc_prefix[RADIX_SIZE];
    for (uint d = 0u; d < RADIX_SIZE; ++d) {
        acc_total[d] = 0u;
        acc_prefix[d] = 0u;
    }

    const uint block_count = get_num_groups(0);
    for (uint b = local_id; b < block_count; b += GROUP_SIZE) {
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            const uint block_sum = block_sums[b * RADIX_SIZE + d];
            acc_total[d] += block_sum;
            if (b < group_id) {
                acc_prefix[d] += block_sum;
            }
        }
    }

    for (uint d = 0u; d < RADIX_SIZE; ++d) {
        thread_totals[local_id][d] = acc_total[d];
        thread_prefixes[local_id][d] = acc_prefix[d];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX_SIZE) {
        const uint chunk_size = GROUP_SIZE / RADIX_SIZE;
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            uint total = 0u;
            uint prefix = 0u;
            for (uint t = local_id * chunk_size; t < (local_id + 1) * chunk_size; ++t) {
                total += thread_totals[t][d];
                prefix += thread_prefixes[t][d];
            }
            chunk_totals[local_id][d] = total;
            chunk_prefixes[local_id][d] = prefix;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0u) {
        uint digit_total[RADIX_SIZE];
        uint digit_prefix[RADIX_SIZE];
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            digit_total[d] = 0u;
            digit_prefix[d] = 0u;
            for (uint chunk = 0u; chunk < RADIX_SIZE; ++chunk) {
                digit_total[d] += chunk_totals[chunk][d];
                digit_prefix[d] += chunk_prefixes[chunk][d];
            }
        }

        uint running = 0u;
        for (uint d = 0u; d < RADIX_SIZE; ++d) {
            digit_bases[d] = running + digit_prefix[d];
            running += digit_total[d];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint column_length = GROUP_SIZE / RADIX_SIZE;
    for (uint stride = 1u; stride < column_length; stride <<= 1) {
        const uint prev = (local_id / RADIX_SIZE >= stride) ? slice_counts[local_id - stride * RADIX_SIZE] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        slice_counts[local_id] += prev;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (id < m) {
        const uint digit = id & (RADIX_SIZE - 1);
        offsets[id] = digit_bases[digit] + (slice_counts[local_id] - value);
    }
}
