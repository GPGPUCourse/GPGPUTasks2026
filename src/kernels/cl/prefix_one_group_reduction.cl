#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define BUF_SIZE (GROUP_SIZE / WARP_SIZE)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_one_group_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    __local unsigned int buf[BUF_SIZE];
    const unsigned int lid = get_local_id(0);
    const unsigned int lane = lid & (WARP_SIZE - 1);
    const unsigned int warp = lid >> WARP_SHIFT;

    unsigned int x = get_global_id(0);
    unsigned int sum;

#if USE_SUBGROUPS
    sum = sub_group_reduce_add(pow2_sum[x]);
#else
    __local unsigned int cache[GROUP_SIZE];
    cache[lid] = pow2_sum[x];
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            cache[warp * WARP_SIZE + lane] += cache[warp * WARP_SIZE + lane + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = cache[warp * WARP_SIZE];
#endif

    x >>= WARP_SHIFT;
    if (lane == 0) {
        buf[warp] = sum;
        pow2_sum[to + x] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int sum = buf[0];
        for (int i = 1; i < BUF_SIZE; ++i) {
            sum += buf[i];
            buf[i] = sum;
        }
    }
}
