#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* output,
    __global       uint* blocks,
    unsigned int n)
{
    const uint index = get_local_id(0);
    const uint global_index = get_global_id(0);

    __local uint temp[GROUP_SIZE];

    const uint value = global_index < n ? input[global_index] : 0;
    temp[index] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    // up sweep
    for (uint i = 1; i < GROUP_SIZE; i *= 2) {
        uint j = (index + 1) * 2 * i - 1;

        if (j < GROUP_SIZE) {
            temp[j] += temp[j-i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index == 0) {
        blocks[get_group_id(0)] = temp[GROUP_SIZE-1];
        temp[GROUP_SIZE-1] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // down sweep
    for (uint i = GROUP_SIZE / 2; i > 0; i /= 2) {
        uint j = (index + 1) * 2 * i - 1;

        if (j < GROUP_SIZE) {
            uint old = temp[j-i];
            temp[j-i] = temp[j];
            temp[j] += old;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_index < n) {
        output[global_index] = temp[index] + value;
    }
}
