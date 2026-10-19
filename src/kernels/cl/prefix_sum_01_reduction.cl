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
    const uint id = get_local_id(0);
    const size_t i = get_global_id(0);
    const uint value = i < n ? input[i] : 0;
    __local uint block[GROUP_SIZE];
    block[id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint lvl = 1; lvl < GROUP_SIZE; lvl *= 2) {
        uint index = (id + 1) * 2 * lvl - 1;
        if (index < GROUP_SIZE)
            block[index] += block[index - lvl];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id == 0) {
        block_sums[get_group_id(0)] = block[GROUP_SIZE - 1];
        block[GROUP_SIZE - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint lvl = GROUP_SIZE / 2; lvl > 0; lvl /= 2) {
        uint index = (id + 1) * 2 * lvl - 1;
        if (index < GROUP_SIZE) {
            uint left = block[index - lvl];
            block[index - lvl] = block[index];
            block[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n)
        output[i] = block[id] + value;
}
