#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_down_scan(
    __global uint* pow2_sum,
    unsigned int cur_pow2,
    unsigned int n
    )
{
    __local uint temp[2 * GROUP_SIZE];
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t block_size = 2 * group_size;
    size_t block_start = get_group_id(0) * block_size;
    
    size_t tree_size = min(block_size, (size_t)n / cur_pow2 - block_start);

    size_t x1 = (block_start + local_id + 1) * cur_pow2 - 1;
    size_t x2 = x1 + group_size * cur_pow2;
    temp[local_id] = x1 < n ? pow2_sum[x1] : 0;
    temp[local_id + group_size] = x2 < n ? pow2_sum[x2] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint cur = tree_size / 2; cur > 0; cur >>= 1) {
        size_t right = (local_id + 1) * (2 * cur) - 1;
        if (right < tree_size) {
            uint left_sum = temp[right - cur];
            uint prefix = temp[right];
            temp[right - cur] = prefix;
            temp[right] = prefix + left_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x1 < n) {
        pow2_sum[x1] = temp[local_id];
    }
    if (x2 < n) {
        pow2_sum[x2] = temp[local_id + group_size];
    }
}
