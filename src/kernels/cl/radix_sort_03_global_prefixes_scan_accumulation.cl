#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define TILE_ROWS (2 * GROUP_SIZE)
#define TILE_SIZE (TILE_ROWS * RADIX) 

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* a,
    __global       uint* group_sums,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint tile_begin = group_id * TILE_SIZE;

    uint offset = 1;
    __local uint buffer[TILE_SIZE];
//    __local uint totals[RADIX];
    uint total = 0;

    for (uint k = local_id; k < TILE_SIZE; k += GROUP_SIZE) {
        const uint g = tile_begin + k;
        buffer[k] = (g < n) ? a[g] : 0u;
    }

    for (uint d = TILE_ROWS / 2; d > 0; d /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = local_id; k < d * RADIX; k += GROUP_SIZE) {
            const uint node  = k / RADIX;
            const uint digit = k % RADIX;
            const uint ind1 = (offset * (2 * node + 1) - 1) * RADIX + digit;
            const uint ind2 = (offset * (2 * node + 2) - 1) * RADIX + digit;
            buffer[ind2] += buffer[ind1];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint digit = local_id; digit < RADIX; digit += GROUP_SIZE) {
        const uint last = (TILE_ROWS - 1) * RADIX + digit;
        total = buffer[last];
//        totals[digit] = total;
        group_sums[group_id * RADIX + digit] = total;
        buffer[last] = 0;
    }

    for (uint d = 1; d < TILE_ROWS; d *= 2) {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = local_id; k < d * RADIX; k += GROUP_SIZE) {
            const uint node  = k / RADIX;
            const uint digit = k % RADIX;
            const uint ind1 = (offset * (2 * node + 1) - 1) * RADIX + digit;
            const uint ind2 = (offset * (2 * node + 2) - 1) * RADIX + digit;
            const uint temp = buffer[ind1];
            buffer[ind1] = buffer[ind2];
            buffer[ind2] += temp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint digit = local_id; digit < RADIX; digit += GROUP_SIZE) {
        buffer[digit] = total;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = local_id; k < TILE_SIZE; k += GROUP_SIZE) {
        const uint g = tile_begin + k;
        if (g < n) {
            const uint row = k / RADIX;
            prefix_sum_accum[g] = (row + 1 < TILE_ROWS) ? buffer[k + RADIX] : buffer[k % RADIX];
        }
    }
}
