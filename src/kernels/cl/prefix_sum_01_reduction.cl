#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* prefix,
    __global       uint* group_sums,
    unsigned int n)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint value = global_id < n ? input[global_id] : 0u;
    __local uint values[GROUP_SIZE];
    values[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = 1; step < GROUP_SIZE; step <<= 1) {
        const uint right = (local_id + 1) * (step << 1) - 1;
        if (right < GROUP_SIZE)
            values[right] += values[right - step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        group_sums[get_group_id(0)] = values[GROUP_SIZE - 1];
        values[GROUP_SIZE - 1] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = GROUP_SIZE >> 1; step > 0; step >>= 1) {
        const uint right = (local_id + 1) * (step << 1) - 1;
        if (right < GROUP_SIZE) {
            const uint left_value = values[right - step];
            values[right - step] = values[right];
            values[right] += left_value;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n)
        prefix[global_id] = values[local_id] + value;
}
