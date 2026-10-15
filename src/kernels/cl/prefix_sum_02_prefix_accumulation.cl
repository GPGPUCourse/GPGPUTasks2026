#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int pow2_offset,
    unsigned int step,
    unsigned int n_half,
    unsigned int max_valid_size)
{
    size_t id = get_global_id(0);
    if (id >= n_half) {
        return;
    }

    unsigned int pair_idx = id / step;
    unsigned int real_id = id + (pair_idx + 1) * step;
    unsigned int sum_idx = pow2_offset + 2 * pair_idx;
    
    uint val = 0;
    if (sum_idx < max_valid_size) {
        val = pow2_sum[sum_idx];
    }

    if (real_id < (n_half * 2)) {
        prefix_sum_accum[real_id] += val;
    }
}