#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#if USE_SUBGROUPS
#  pragma OPENCL EXTENSION cl_khr_subgroup : enable
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    unsigned int x = get_global_id(0) & ~(GROUP_SIZE - 1);
    unsigned int from = 0;
    unsigned int to = n;
    unsigned int sum = 0;

    __local unsigned int cache[GROUP_SIZE];
    const unsigned int lid = get_local_id(0);
    const unsigned int lane = lid & (GROUP_SIZE - 1);
    const unsigned int group = lid >> GROUP_SHIFT;

    cache[lid] = pow2_sum[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        unsigned int v = 0;
        if (lane >= offset) {
            v = cache[group * GROUP_SIZE + lane - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane >= offset) {
            cache[group * GROUP_SIZE + lane] += v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = cache[lid];

    from = to;
    n >>= GROUP_SHIFT;
    to = from + n;
    x >>= GROUP_SHIFT;

    #pragma unroll
    for (int i = GROUP_SHIFT; i < 32; ++i) {
        if (x & 1) {
            sum += pow2_sum[from + x - 1];
        }
        x >>= 1;
        from = to;
        n = (n + 1) / 2;
        to += n;
    }
    prefix_sum_accum[get_global_id(0)] = sum;
}
