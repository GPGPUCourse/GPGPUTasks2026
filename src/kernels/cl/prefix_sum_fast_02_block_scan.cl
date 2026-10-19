#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_fast_02_block_scan(
    __global const uint* input,
    __global       uint* output,
    __global const uint* block_offsets)
{
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint base = group * SCAN_BLOCK;

    __local uint data[SCAN_BLOCK];
    __local uint thread_sums[GROUP_SIZE];

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        data[j * GROUP_SIZE + lid] = input[base + j * GROUP_SIZE + lid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint first = lid * ELEMS_PER_THREAD;

    uint sum = 0;

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        sum += data[first + j];
    }

    thread_sums[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        uint add = 0;

        if (lid >= offset) {
            add = thread_sums[lid - offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid >= offset) {
            thread_sums[lid] += add;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint prefix = 0;

    if (group > 0) {
        prefix = block_offsets[group - 1];
    }

    if (lid > 0) {
        prefix += thread_sums[lid - 1];
    }

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        prefix += data[first + j];
        data[first + j] = prefix;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        output[base + j * GROUP_SIZE + lid] = data[j * GROUP_SIZE + lid];
    }
}