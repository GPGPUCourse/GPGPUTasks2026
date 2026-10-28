#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// каждый поток берет непрерывный кусок из ELEMS_PER_THREAD элементов - так порядок внутри цифры сохраняется (сортировка стабильная)
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
    const uint first = g * SCAN_BLOCK + lid * ELEMS_PER_THREAD;
    __local uint pos[RADIX][GROUP_SIZE];

    for (uint b = 0; b < RADIX; ++b)
        pos[b][lid] = 0;
    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        const uint i = first + j;
        if (i < n)
            ++pos[(in[i] >> shift) & (RADIX - 1)][lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX) {
        uint s = offsets[lid * n_groups + g];
        for (uint t = 0; t < GROUP_SIZE; ++t) {
            const uint c = pos[lid][t];
            pos[lid][t] = s;
            s += c;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        const uint i = first + j;
        if (i < n) {
            const uint x = in[i];
            out[pos[(x >> shift) & (RADIX - 1)][lid]++] = x;
        }
    }
}
