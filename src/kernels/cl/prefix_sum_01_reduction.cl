#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* output,
    __global uint* block_sums,
    unsigned int n)
{
    const uint tid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint i = block * GROUP_SIZE + tid;

    __local uint values[GROUP_SIZE];

    values[tid] = i < n ? input[i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        uint sum = values[tid];

        if (tid >= offset)
            sum += values[tid - offset];

        barrier(CLK_LOCAL_MEM_FENCE);
        values[tid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n)
        output[i] = values[tid];

    if (tid == GROUP_SIZE - 1)
        block_sums[block] = values[tid];
}