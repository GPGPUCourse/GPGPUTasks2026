#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* block_prefix_sum,
    __global uint* output,
    unsigned int n)
{
    const size_t i = get_global_id(0);
    const size_t block = get_group_id(0);
    if (i < n && block > 0)
        output[i] += block_prefix_sum[block - 1];
}
