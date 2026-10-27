#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_put(
    __global uint* input,
    __global uint* output,
    __global uint* pow2_sum,
    __global uint* group_accum,
    __global uint* start_loc, // one value, not array
    unsigned int n,
    unsigned int iter,
    unsigned int mask)
{
    __local unsigned int cache[GROUP_SIZE];
    unsigned int sum = 0;
    unsigned int gid = get_global_id(0);
    {
        unsigned int x = get_global_id(0) & ~(GROUP_SIZE - 1);
        unsigned int from = 0;
        unsigned int to = n;
        unsigned int num = to - from;

        const unsigned int lid = get_local_id(0);
        const unsigned int lane = lid & (GROUP_SIZE - 1);
        const unsigned int group = lid >> GROUP_SHIFT;

        sum += group_accum[get_group_id(0)];

        cache[lid] = (gid < n ? pow2_sum[gid] : 0);
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
        sum += cache[lid];

        from = to;
        num >>= GROUP_SHIFT;
        to = from + num;
        x >>= GROUP_SHIFT;

#pragma unroll
        for (int i = GROUP_SHIFT; i < 32; ++i) {
            if (x & 1) {
                sum += pow2_sum[from + x - 1];
            }
            x >>= 1;
            from = to;
            num = (num + 1) / 2;
            to += num;
        }
    }

    unsigned int x = gid;
    if (x < n) {
        unsigned int marker = (input[x] >> (iter * RADIX_SHIFT)) & ((1 << RADIX_SHIFT) - 1);
        unsigned int loc = sum;
        unsigned int start = start_loc[mask & 1];
        if (marker == mask) {
            output[start + loc - 1] = input[x];
        }

        if (x == n - 1) {
            if (start + loc >= n) {
                start_loc[(mask + 1) & 1] = 0;
            } else {
                start_loc[(mask + 1) & 1] = start + loc;
            }
        }
    }
}
