#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* counts,
    unsigned int n,
    unsigned int shift)
{
    __local uint hist[RADIX_BINS];
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);

    if (lid < RADIX_BINS) {
        hist[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n) {
        atomic_inc(&hist[(input[i] >> shift) & (RADIX_BINS - 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BINS) {
        counts[lid * get_num_groups(0) + get_group_id(0)] = hist[lid];
    }
}
