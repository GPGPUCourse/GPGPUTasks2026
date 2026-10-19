#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global uint* next_pow2_sum, // will contain (n+1)/2 values
    uint n,
    uint pow2_sum_begin,
    uint next_pow2_sum_begin)
{
    uint i = get_global_id(0), sum = 0;
    if (2 * i + 1 < n) {
        next_pow2_sum[next_pow2_sum_begin + i] = pow2_sum[pow2_sum_begin + 2 * i] + pow2_sum[pow2_sum_begin + 2 * i + 1];
    }
}