#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* in,
    __global uint* out,
    __global const uint* block_offsets)
{
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);
    const uint first = g * SCAN_BLOCK + lid * ELEMS_PER_THREAD;
    __local uint sums[GROUP_SIZE];

    uint sum = 0;
    for (uint j = 0; j < ELEMS_PER_THREAD; ++j)
        sum += in[first + j];
    sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint s = block_offsets[g];
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            const uint t = sums[i];
            sums[i] = s;
            s += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint s = sums[lid];
    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        s += in[first + j];
        out[first + j] = s;
    }
}