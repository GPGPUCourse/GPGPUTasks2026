#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

inline uint digit(uint x, uint shift)
{
    return (x >> shift) & (RADIX - 1);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint* in,
    __global const uint* offsets,
    __global uint* out,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);
    const uint n_groups = get_num_groups(0);
    const uint base = g * SCAN_BLOCK;
    const uint size = min((uint)SCAN_BLOCK, n - base);
    const uint first = lid * ELEMS_PER_THREAD;
    __local uint data[SCAN_BLOCK];
    __local uint sorted[SCAN_BLOCK];
    __local uint pos[RADIX][GROUP_SIZE];
    __local uint start[RADIX];

    for (uint k = lid; k < size; k += GROUP_SIZE)
        data[k] = in[base + k];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint cnt[RADIX] = { 0 };
    for (uint k = first; k < min(first + ELEMS_PER_THREAD, size); ++k)
        ++cnt[digit(data[k], shift)];
    for (uint b = 0; b < RADIX; ++b)
        pos[b][lid] = cnt[b];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint off = 1; off < GROUP_SIZE; off <<= 1) {
        uint t[RADIX];
        for (uint b = 0; b < RADIX; ++b)
            t[b] = lid >= off ? pos[b][lid - off] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint b = 0; b < RADIX; ++b)
            pos[b][lid] += t[b];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        uint s = 0;
        for (uint b = 0; b < RADIX; ++b) {
            start[b] = s;
            s += pos[b][GROUP_SIZE - 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = 0; b < RADIX; ++b)
        cnt[b] = start[b] + pos[b][lid] - cnt[b];
    for (uint k = first; k < min(first + ELEMS_PER_THREAD, size); ++k)
        sorted[cnt[digit(data[k], shift)]++] = data[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = lid; k < size; k += GROUP_SIZE) {
        const uint d = digit(sorted[k], shift);
        out[offsets[d * n_groups + g] + k - start[d]] = sorted[k];
    }
}
