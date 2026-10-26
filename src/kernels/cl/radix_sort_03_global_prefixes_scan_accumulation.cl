#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global uint* histogram_accum,
    __global const uint* histogram_group_sums,
    unsigned int n
) {
    const uint idx = get_global_id(0);
    const uint id = idx / RADIX;

    const uint block_id = id / (GROUP_SIZE * 2);
    const uint d = idx % RADIX;

    if (idx < n && block_id > 0) {
        histogram_accum[idx] += histogram_group_sums[(block_id - 1) * RADIX + d];
    }
}
