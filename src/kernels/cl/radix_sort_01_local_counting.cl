#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* keys,
    __global uint*       hist,        // [RADIX * num_tiles], digit-major
    uint n, uint shift, uint num_tiles)
{
    const uint lid  = get_local_id(0);
    const uint tile = get_group_id(0);

    __local uint local_hist[RADIX];
    if (lid < RADIX)
        local_hist[lid] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint base = tile * GROUP_SIZE * RADIX_ITEMS;
    for (uint j = 0; j < RADIX_ITEMS; ++j) {
        const uint i = base + j * GROUP_SIZE + lid;   // coalesced; order doesn't matter for counting
        if (i < n)
            atomic_inc(&local_hist[(keys[i] >> shift) & (RADIX - 1u)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX)
        hist[lid * num_tiles + tile] = local_hist[lid];
}