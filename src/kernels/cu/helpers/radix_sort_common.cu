#pragma once

#include "../../defines.h"

#define FULL_MASK 0xffffffffu
#define NWARPS (GROUP_SIZE / 32)

static_assert(GROUP_SIZE == RADIX, "one thread per digit is assumed");
static_assert(RADIX_ITEMS_PER_THREAD % 4 == 0, "uint4 loads are used");

__device__ __forceinline__ unsigned int radix_digit(unsigned int key, unsigned int shift)
{
    return (key >> shift) & RADIX_MASK;
}

__device__ __forceinline__ unsigned int block_exclusive_scan(unsigned int x, unsigned int& total, unsigned int* s_warp_sums)
{
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;

    unsigned int inclusive = x;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned int y = __shfl_up_sync(FULL_MASK, inclusive, offset);
        if (lane >= offset)
            inclusive += y;
    }
    if (lane == 31)
        s_warp_sums[warp] = inclusive;
    __syncthreads();

    unsigned int warp_prefix = 0;
    total = 0;
    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w) {
        unsigned int v = s_warp_sums[w];
        if (w < warp)
            warp_prefix += v;
        total += v;
    }
    __syncthreads();
    return warp_prefix + inclusive - x;
}
