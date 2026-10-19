#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const size_t index = get_global_id(0);
    const size_t left = 2 * index;
    if (left >= n)
        return;

    uint sum = pow2_sum[left];
    if (left + 1 < n)
        sum += pow2_sum[left + 1];
    next_pow2_sum[index] = sum;
}
