#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFS(i) ((i) >> LOG_NUM_BANKS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* prefix_sum,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[2 * GROUP_SIZE + CONFLICT_FREE_OFFS(2 * GROUP_SIZE)];
    unsigned int tid = get_local_id(0);
    unsigned int first_local = tid + CONFLICT_FREE_OFFS(tid);
    unsigned int second_local = tid + GROUP_SIZE + CONFLICT_FREE_OFFS(tid + GROUP_SIZE);
    size_t first = get_group_id(0) * (2 * GROUP_SIZE) + tid;
    size_t second = first + GROUP_SIZE;
    uint first_value = first < n ? input[first] : 0;
    uint second_value = second < n ? input[second] : 0;
    temp[first_local] = first_value;
    temp[second_local] = second_value;

    unsigned int offset = 1;
    for (unsigned int d = GROUP_SIZE; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFS(ai);
            bi += CONFLICT_FREE_OFFS(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        unsigned int last = 2 * GROUP_SIZE - 1 + CONFLICT_FREE_OFFS(2 * GROUP_SIZE - 1);
        block_sums[get_group_id(0)] = temp[last];
        temp[last] = 0;
    }

    for (unsigned int d = 1; d < 2 * GROUP_SIZE; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFS(ai);
            bi += CONFLICT_FREE_OFFS(bi);
            uint left_sum = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += left_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (first < n) {
        prefix_sum[first] = temp[first_local] + first_value;
    }
    if (second < n) {
        prefix_sum[second] = temp[second_local] + second_value;
    }
}
