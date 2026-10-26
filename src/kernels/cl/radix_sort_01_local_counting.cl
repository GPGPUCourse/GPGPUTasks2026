#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX 16
#define TILE (4 * GROUP_SIZE)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* counts,
    unsigned int n,
    unsigned int shift)
{
    __local uint hist[RADIX];
    unsigned int tid = get_local_id(0);
    unsigned int base = get_group_id(0) * TILE;
    unsigned int end = min(base + TILE, n);

    if (tid < RADIX) {
        hist[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = base + tid; i < end; i += GROUP_SIZE) {
        atomic_inc(&hist[(input[i] >> shift) & (RADIX - 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < RADIX) {
        counts[tid * get_num_groups(0) + get_group_id(0)] = hist[tid];
    }
}
