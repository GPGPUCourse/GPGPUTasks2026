#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* output,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[PREFIX_SUM_BLOCK_SIZE + CONFLICT_FREE_OFFSET(PREFIX_SUM_BLOCK_SIZE - 1)];

    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int block_start = group_id * PREFIX_SUM_BLOCK_SIZE;

    const unsigned int a_index = local_id;
    const unsigned int b_index = local_id + GROUP_SIZE;
    const uint a = (block_start + a_index < n) ? input[block_start + a_index] : 0;
    const uint b = (block_start + b_index < n) ? input[block_start + b_index] : 0;
    temp[a_index + CONFLICT_FREE_OFFSET(a_index)] = a;
    temp[b_index + CONFLICT_FREE_OFFSET(b_index)] = b;

    unsigned int offset = 1;
    for (unsigned int d = PREFIX_SUM_BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            unsigned int ai = offset * (2 * local_id + 1) - 1;
            unsigned int bi = offset * (2 * local_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (local_id == 0) {
        const unsigned int last = PREFIX_SUM_BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(PREFIX_SUM_BLOCK_SIZE - 1);
        block_sums[group_id] = temp[last];
        temp[last] = 0;
    }

    for (unsigned int d = 1; d < PREFIX_SUM_BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < d) {
            unsigned int ai = offset * (2 * local_id + 1) - 1;
            unsigned int bi = offset * (2 * local_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            const uint t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (block_start + a_index < n)
        output[block_start + a_index] = temp[a_index + CONFLICT_FREE_OFFSET(a_index)] + a;
    if (block_start + b_index < n)
        output[block_start + b_index] = temp[b_index + CONFLICT_FREE_OFFSET(b_index)] + b;
}
