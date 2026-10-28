#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

inline uint digit(uint x, uint shift)
{
    return (x >> shift) & (RADIX - 1);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* in,
    __global uint* counts,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);
    const uint n_groups = get_num_groups(0);
    const uint base = g * SCAN_BLOCK;
    const uint size = min((uint)SCAN_BLOCK, n - base);
    __local uint hist[RADIX];

    if (lid < RADIX)
        hist[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = lid; k < size; k += GROUP_SIZE)
        atomic_inc(&hist[digit(in[base + k], shift)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX)
        counts[lid * n_groups + g + 1] = hist[lid];
}
