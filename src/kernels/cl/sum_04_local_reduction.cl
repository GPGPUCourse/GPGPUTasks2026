#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);

    __local uint local_data[GROUP_SIZE];

    if (gid < n) {
        local_data[lid] = a[gid];
    } else {
        local_data[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
        if (lid < offset) {
            local_data[lid] += local_data[lid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        b[group] = local_data[0];
    }
}
