#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_sweep_up(
    // АНТИХАЙП!11!!1
    __global const uint* input, // contains n values
    __global       uint* output_all, // will contain n values
    __global       uint* reducted, // will contain (n + GROUP_SIZE - 1) / GROUP_SIZE values
    unsigned int n)
{
    const uint idx = get_global_id(0);
    const uint block_idx = get_local_id(0);

    __local uint buff[GROUP_SIZE];
    buff[block_idx] = idx < n ? input[idx] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint level = 1; level < GROUP_SIZE; level *= 2) {
        const uint delta = level <= block_idx ? buff[block_idx - level] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        buff[block_idx] += delta;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx < n) {
        output_all[idx] = buff[block_idx];
    }
    if (block_idx == 0) {
        reducted[get_group_id(0)] = buff[GROUP_SIZE - 1];
    }
}
