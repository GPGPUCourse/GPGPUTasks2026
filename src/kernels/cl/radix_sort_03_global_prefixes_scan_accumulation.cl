#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Exclusive prefix scan of the flattened digit-major histogram.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sums,
    __global uint* global_prefix,
    uint count,
    uint pow2)
{
    const uint i = get_global_id(0);
    if (i >= count) return;
    const uint block_size = 1u << pow2;
    if (i & block_size)
        global_prefix[i] += pow2_sums[i / block_size - 1u];
}
