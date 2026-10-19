#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* a,
    __global       uint* group_sums,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    const uint id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);

    uint offset = 1;
    __local uint buffer[2 * GROUP_SIZE + 1];

    if (id * 2 < n) {
        buffer[2 * local_id] = a[2 * id];
    }
    else {
        buffer[2 * local_id] = 0;
    }

    if (id * 2 + 1 < n) {
        buffer[2 * local_id + 1] = a[2 * id + 1];
    }
    else {
        buffer[2 * local_id + 1] = 0;
    }

    for (uint d = GROUP_SIZE; d > 0; d /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            buffer[offset * (2 * local_id + 2) - 1] += buffer[offset * (2 * local_id + 1) - 1];
        }
        offset *= 2;
    }

    if (local_id == 0) {
        buffer[2 * GROUP_SIZE] = buffer[2 * GROUP_SIZE - 1];
        buffer[2 * GROUP_SIZE - 1] = 0;

        group_sums[group_id] = buffer[2 * GROUP_SIZE];
    }

    for (uint d = 1; d < 2 * GROUP_SIZE; d *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset /= 2;
        if (local_id < d) {
            const uint ind1 = offset * (2 * local_id + 1) - 1;
            const uint ind2 = offset * (2 * local_id + 2) - 1;
            const uint temp = buffer[ind1];
            buffer[ind1] = buffer[ind2];
            buffer[ind2] += temp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (2 * id < n) {
        prefix_sum_accum[2 * id] = buffer[2 * local_id + 1];
    }

    if (2 * id + 1 < n) {
        prefix_sum_accum[2 * id + 1] = buffer[2 * local_id + 2];
    }
}
