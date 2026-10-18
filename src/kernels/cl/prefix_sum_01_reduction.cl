#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global uint* next_pow2_sum,
    const uint n)
{
    const uint i = get_global_id(0);
    const uint pairs = n >> 1u;

    if (i < pairs) {
        const uint2 v = vload2(i, pow2_sum);
        next_pow2_sum[i] = v.x + v.y;
    } else if ((n & 1u) && i == pairs) {
        next_pow2_sum[i] = pow2_sum[n - 1u];
    }
}