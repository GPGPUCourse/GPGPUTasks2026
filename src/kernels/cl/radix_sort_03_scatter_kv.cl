#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_03_scatter_kv(
    __global const uint* keys_in,
    __global const uint* vals_in,
    __global const uint* hist, // scanned
    __global uint* keys_out,
    __global uint* vals_out,
    uint n, uint shift, uint num_tiles)
{
    const uint lid = get_local_id(0);
    const uint tile = get_group_id(0);

    __local uint cnt[RADIX * GROUP_SIZE]; // [digit][work-item]
    __local uint tile_off[RADIX];

    if (lid < RADIX)
        tile_off[lid] = hist[lid * num_tiles + tile];

    const uint base = (tile * GROUP_SIZE + lid) * RADIX_ITEMS;

    uint k[RADIX_ITEMS], v[RADIX_ITEMS];
    uint mine[RADIX];
#pragma unroll
    for (uint d = 0; d < RADIX; ++d)
        mine[d] = 0u;

#pragma unroll
    for (uint j = 0; j < RADIX_ITEMS; ++j) {
        if (base + j < n) {
            k[j] = keys_in[base + j];
            v[j] = vals_in[base + j];
            mine[(k[j] >> shift) & (RADIX - 1u)]++;
        }
    }

#pragma unroll
    for (uint d = 0; d < RADIX; ++d)
        cnt[d * GROUP_SIZE + lid] = mine[d];
    barrier(CLK_LOCAL_MEM_FENCE);

    // inclusive scan over work-items, separately for every digit
    for (uint off = 1; off < GROUP_SIZE; off <<= 1) {
        uint t[RADIX];
#pragma unroll
        for (uint d = 0; d < RADIX; ++d)
            t[d] = lid >= off ? cnt[d * GROUP_SIZE + lid - off] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (uint d = 0; d < RADIX; ++d)
            cnt[d * GROUP_SIZE + lid] += t[d];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // exclusive position of my first key of each digit within the tile
#pragma unroll
    for (uint d = 0; d < RADIX; ++d)
        mine[d] = cnt[d * GROUP_SIZE + lid] - mine[d];

#pragma unroll
    for (uint j = 0; j < RADIX_ITEMS; ++j) {
        if (base + j < n) {
            const uint d = (k[j] >> shift) & (RADIX - 1u);
            const uint dst = tile_off[d] + mine[d]++;
            keys_out[dst] = k[j];
            vals_out[dst] = v[j];
        }
    }
}