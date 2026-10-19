#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global uint* prefix_sum_accum,
    __global uint* block_sums,
    unsigned int n,
    unsigned int pow2)
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    __local uint tile[GROUP_SIZE];


    if (pow2 == 0) {

        uint value = (gid < n) ? pow2_sum[gid] : 0;
        tile[lid] = value;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            uint add = (lid >= offset) ? tile[lid - offset] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            tile[lid] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (gid < n)
            prefix_sum_accum[gid] = tile[lid];

        if (lid == GROUP_SIZE - 1)
            block_sums[get_group_id(0)] = tile[lid];
    } else {
        if (gid < n) {
            uint value = pow2_sum[gid];
            uint block = gid / pow2;

            if (block > 0)
                value += block_sums[block - 1];

            prefix_sum_accum[gid] = value;
        }
    }
}
