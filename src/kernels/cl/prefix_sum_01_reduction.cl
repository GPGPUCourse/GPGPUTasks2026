#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFS(i) ((i) >> LOG_NUM_BANKS)

#define VECTOR_WIDTH 4
#define VECTORS_PER_THREAD 2
#define ELEMENTS_PER_THREAD (VECTOR_WIDTH * VECTORS_PER_THREAD)
#define SCAN_BLOCK_SIZE (GROUP_SIZE * ELEMENTS_PER_THREAD)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* prefix_sum,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[GROUP_SIZE + CONFLICT_FREE_OFFS(GROUP_SIZE)];
    unsigned int tid = get_local_id(0);
    unsigned int local_index = tid + CONFLICT_FREE_OFFS(tid);
    size_t thread_start = get_group_id(0) * SCAN_BLOCK_SIZE + ELEMENTS_PER_THREAD * tid;
    uint4 values[VECTORS_PER_THREAD];
    uint thread_sum = 0;
    #pragma unroll
    for (unsigned int part = 0; part < VECTORS_PER_THREAD; ++part) {
        size_t first = thread_start + VECTOR_WIDTH * part;
        values[part] = 0;
        for (unsigned int i = 0; i < VECTOR_WIDTH; ++i) {
            if (first + i < n) {
                values[part][i] = input[first + i];
            }
        }

        values[part].s1 += values[part].s0;
        values[part].s2 += values[part].s1;
        values[part].s3 += values[part].s2;
        thread_sum += values[part].s3;
    }
    temp[local_index] = thread_sum;

    unsigned int offset = 1;
    for (unsigned int d = GROUP_SIZE / 2; d > 0; d >>= 1) {
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
        unsigned int last = GROUP_SIZE - 1 + CONFLICT_FREE_OFFS(GROUP_SIZE - 1);
        block_sums[get_group_id(0)] = temp[last];
        temp[last] = 0;
    }

    for (unsigned int d = 1; d < GROUP_SIZE; d <<= 1) {
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

    uint prefix = temp[local_index];
    #pragma unroll
    for (unsigned int part = 0; part < VECTORS_PER_THREAD; ++part) {
        values[part] += prefix;
        prefix = values[part].s3;
        size_t first = thread_start + VECTOR_WIDTH * part;
        for (unsigned int i = 0; i < VECTOR_WIDTH; ++i) {
            if (first + i < n) {
                prefix_sum[first + i] = values[part][i];
            }
        }
    }
}
