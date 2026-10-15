#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_sweep_down(
    // АНТИХАЙП!11!!1
    __global const uint* input,
    __global       uint* output,
    unsigned int n)
{
    const uint idx = get_global_id(0);
    const uint block_id = get_group_id(0);

    if (idx >= n || block_id == 0) {
        return;
    }

    output[idx] += input[block_id - 1];
}
