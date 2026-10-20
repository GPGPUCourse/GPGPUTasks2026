#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,      // n значений: суммы блоков длины 2^pow2
    __global       uint* next_pow2_sum, // (n+1)/2 значений: суммы блоков длины 2^(pow2+1)
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= (n + 1) / 2)
        return;

    const unsigned int left = pow2_sum[2 * i];
    const unsigned int right = (2 * i + 1 < n) ? pow2_sum[2 * i + 1] : 0;
    next_pow2_sum[i] = left + right;
}
