#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* in,
    __global uint* block_sums,
    unsigned int n)
{
    const uint first = get_group_id(0) * SCAN_BLOCK + get_local_id(0);
    uint sum = 0;
    for (uint j = 0; j < ELEMS_PER_THREAD; ++j) {
        const uint i = first + j * GROUP_SIZE;
        if (i < n)
            sum += in[i];
    }
    atomic_add(&block_sums[get_group_id(0) + 1], sum);
}
