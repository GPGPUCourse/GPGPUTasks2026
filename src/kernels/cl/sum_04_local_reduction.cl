#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(
    __global const uint* a,
    __global uint* b,
    unsigned int n)
{
    const uint global_idx = get_global_id(0);
    const uint local_idx = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    if (global_idx < n) {
        local_data[local_idx] = a[global_idx];
    } else {
        local_data[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            local_data[local_idx] += local_data[local_idx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_idx == 0) {
        b[get_group_id(0)] = local_data[0];
    }
}
