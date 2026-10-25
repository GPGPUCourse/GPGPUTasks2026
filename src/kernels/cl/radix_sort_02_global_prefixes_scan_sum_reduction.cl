#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global uint* prefixes,
    __global uint* sums,
    unsigned int n)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    
    __local uint scan[GROUP_SIZE];
    scan[local_id] = global_id < n
        ? prefixes[global_id]
        : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        uint index = (local_id + 1) * 2 * stride - 1;

        if (index < GROUP_SIZE) {
            scan[index] += scan[index - stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        sums[get_group_id(0)] = scan[GROUP_SIZE - 1];
        scan[GROUP_SIZE - 1] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        uint index = (local_id + 1) * 2 * stride - 1;

        if (index < GROUP_SIZE) {
            uint left = scan[index - stride];
            scan[index - stride] = scan[index];
            scan[index] += left;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n) {
        prefixes[global_id] = scan[local_id];
    }
}
