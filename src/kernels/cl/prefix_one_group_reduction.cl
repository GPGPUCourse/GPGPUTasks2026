#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define BUF_SIZE (GROUP_SIZE / WARP_SIZE)

#if USE_SUBGROUPS
#  pragma OPENCL EXTENSION cl_khr_subgroup : enable
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_one_group_reduction(
    __global uint* pow2_sum,
    __global const uint* input,
    unsigned int iter, unsigned int mask,
    unsigned int from, unsigned int to)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int lane = lid & (GROUP_SIZE - 1);
    const unsigned int group = lid >> GROUP_SHIFT;

    const unsigned int count = to - from;
    unsigned int x = get_global_id(0);
    unsigned int v = 0;
    if (x < count) {
        unsigned int marker = (input[x] >> (iter * RADIX_SHIFT)) & ((1 << RADIX_SHIFT) - 1);
        v = (marker == mask ? 1 : 0);
        pow2_sum[x] = v;
    }

    unsigned int sum;

    __local unsigned int cache[GROUP_SIZE];
    cache[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (unsigned int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            cache[group * GROUP_SIZE + lane] += cache[group * GROUP_SIZE + lane + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = cache[group * GROUP_SIZE];

    x >>= GROUP_SHIFT;
    if (lane == 0 && x < (count >> GROUP_SHIFT)) {
        pow2_sum[to + x] = sum;
    }
}
