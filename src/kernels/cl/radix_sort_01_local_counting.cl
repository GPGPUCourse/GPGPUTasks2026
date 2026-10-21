#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* output,
    uint length,
    uint digit)
{
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint stage[GROUP_SIZE];

    bool is_bound = i < length;

    stage[lid] = is_bound ? ((input[i] >> digit) & 1u) : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (is_bound) {
        output[i] = stage[lid];
    }
}
