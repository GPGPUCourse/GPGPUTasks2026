#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global uint* prefix_sum_accum,
    const uint n,
    const uint pow2)
{
    const uint i = get_global_id(0);

    const uint block = i >> pow2;
    const uint offset = i & ((1u << pow2) - 1u);
    const uint k = (((block << 1u) | 1u) << pow2) + offset - 1u;

    if (k < n) {
        prefix_sum_accum[k] += pow2_sum[block << 1u];
    }
}