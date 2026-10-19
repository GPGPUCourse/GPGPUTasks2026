#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global uint* next_pow2_sum,
    unsigned int n,
    unsigned int pow2)
{
    const uint i = get_global_id(0);

    if (i < n) {
        uint value = pow2_sum[i];

        if (i >= pow2)
            value += pow2_sum[i - pow2];

        next_pow2_sum[i] = value;
    }
}
