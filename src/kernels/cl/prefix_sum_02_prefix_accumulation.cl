#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

inline uint pad(uint i)
{
    return i + i / 32;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* in,
    __global uint* out,
    __global const uint* block_offsets)
{
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);
    const uint base = g * SCAN_BLOCK;
    const uint first = lid * ELEMS_PER_THREAD;
    __local uint data[SCAN_BLOCK + SCAN_BLOCK / 32];
    __local uint sums[GROUP_SIZE];

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j)
        data[pad(j * GROUP_SIZE + lid)] = in[base + j * GROUP_SIZE + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sum = 0;
    for (uint j = 0; j < ELEMS_PER_THREAD; ++j)
        sum += data[pad(first + j)];
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
        s += data[pad(first + j)];
        data[pad(first + j)] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j)
        out[base + j * GROUP_SIZE + lid] = data[pad(j * GROUP_SIZE + lid)];
}