#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global uint* next_pow2_sum, // will contain (n+1)/2 values
    __global uint* prefix_sum,
    unsigned int n)
{
    const uint lane = get_local_id(0);
    const size_t first = get_group_id(0) * (2 * GROUP_SIZE) + lane;
    const size_t second = first + GROUP_SIZE;

    const uint a = first < n ? pow2_sum[first] : 0;
    const uint b = second < n ? pow2_sum[second] : 0;

    __local uint sums[(2 * GROUP_SIZE)];

    sums[lane] = a;
    sums[lane + GROUP_SIZE] = b;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < (2 * GROUP_SIZE); stride <<= 1) {
        const uint index = (lane + 1) * 2 * stride - 1;
        if (index < (2 * GROUP_SIZE)) {
            sums[index] += sums[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        next_pow2_sum[get_group_id(0)] = sums[(2 * GROUP_SIZE) - 1];
        sums[(2 * GROUP_SIZE) - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = (2 * GROUP_SIZE) / 2; stride > 0; stride >>= 1) {
        const uint index = (lane + 1) * 2 * stride - 1;
        if (index < (2 * GROUP_SIZE)) {
            const uint left = sums[index - stride];
            sums[index - stride] = sums[index];
            sums[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (first < n) {
        prefix_sum[first] = sums[lane] + a;
    }
    if (second < n) {
        prefix_sum[second] = sums[lane + GROUP_SIZE] + b;
    }
}
