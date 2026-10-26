#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* a,
    __global const uint* prefix_histogram,
    __global       uint* sorted,
    unsigned int n,
    unsigned int bit)
{
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    __local uint buffer[GROUP_SIZE * 2];
    __local uint prefix[PAD(GROUP_SIZE * 2)];
    __local uint digit_start[RADIX];
    __local uint global_start[RADIX];
    __local uint local_start[RADIX];
    __local uint total_sum_prev;

    if (local_id < RADIX) {
        digit_start[local_id] = (group_id == 0) ? 0 : prefix_histogram[(group_id - 1) * RADIX + local_id];
        global_start[local_id] = prefix_histogram[(num_groups - 1) * RADIX + local_id];
    }

    if (local_id == 0) {
        total_sum_prev = 0;
    }

    uint offset = 1;

    uint i1 = local_id;
    uint i2 = local_id + GROUP_SIZE;
    uint g1 = group_id * 2 * GROUP_SIZE + local_id;
    uint g2 = g1 + GROUP_SIZE;

    uint v1 = g1 < n ? a[g1] : UINT_MAX;
    uint v2 = g2 < n ? a[g2] : UINT_MAX;
    uint d1 = (v1 >> bit) & (RADIX - 1);
    uint d2 = (v2 >> bit) & (RADIX - 1);

    uint rank1 = 0;
    uint rank2 = 0;

    for (uint i = 0; i < RADIX; i++) {
        barrier(CLK_LOCAL_MEM_FENCE);

        const uint i1_bit = (d1 == i);
        const uint i2_bit = (d2 == i);

        prefix[PAD(i1)] = i1_bit;
        prefix[PAD(i2)] = i2_bit;

        for (uint d = GROUP_SIZE; d > 0; d /= 2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < d) {
                prefix[PAD(offset * (2 * local_id + 2) - 1)] += prefix[PAD(offset * (2 * local_id + 1) - 1)];
            }
            offset *= 2;
        }

        if (local_id == 0) {
            prefix[PAD(2 * GROUP_SIZE - 1)] = 0;
        }

        for (uint d = 1; d < 2 * GROUP_SIZE; d *= 2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset /= 2;
            if (local_id < d) {
                const uint ind1 = offset * (2 * local_id + 1) - 1;
                const uint ind2 = offset * (2 * local_id + 2) - 1;
                const uint temp = prefix[PAD(ind1)];
                prefix[PAD(ind1)] = prefix[PAD(ind2)];
                prefix[PAD(ind2)] += temp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i1_bit) rank1 = prefix[PAD(i1)] + total_sum_prev;
        if (i2_bit) rank2 = prefix[PAD(i2)] + total_sum_prev;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i2 == 2 * GROUP_SIZE - 1) {
            local_start[i] = total_sum_prev;
            total_sum_prev += prefix[PAD(i2)] + (d2 == i);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    buffer[rank1] = v1;
    buffer[rank2] = v2;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = local_id; i < min(2u * GROUP_SIZE, n - group_id * (2 * GROUP_SIZE)); i += GROUP_SIZE) {
        const uint v = buffer[i];
        const uint d = (v >> bit) & (RADIX - 1);
        uint rank = i - local_start[d] + digit_start[d];

        for (uint j = 0; j < d; j++) {
            rank += global_start[j];
        }

        sorted[rank] = v;
    }
}