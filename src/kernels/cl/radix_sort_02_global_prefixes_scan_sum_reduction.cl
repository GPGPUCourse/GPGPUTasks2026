#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global       uint* prefix_sum,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[2 * GROUP_SIZE];
    const uint tid = get_local_id(0);
    const uint first = get_group_id(0) * (2 * GROUP_SIZE) + tid;
    const uint second = first + GROUP_SIZE;
    const uint first_value = first < n ? input[first] : 0;
    const uint second_value = second < n ? input[second] : 0;
    temp[tid] = first_value;
    temp[tid + GROUP_SIZE] = second_value;

    uint offset = 1;
    for (uint d = GROUP_SIZE; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        block_sums[get_group_id(0)] = temp[2 * GROUP_SIZE - 1];
        temp[2 * GROUP_SIZE - 1] = 0;
    }

    for (uint d = 1; d < 2 * GROUP_SIZE; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint left_sum = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += left_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (first < n) {
        prefix_sum[first] = temp[tid] + first_value;
    }
    if (second < n) {
        prefix_sum[second] = temp[tid + GROUP_SIZE] + second_value;
    }
}
