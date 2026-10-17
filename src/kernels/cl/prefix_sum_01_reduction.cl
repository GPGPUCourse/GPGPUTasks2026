#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,  // Иходной массив
    __global       uint* levels, // Уровень 1+
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= (n + 1) / 2)
        return;

    const uint left = input[2 * i];
    const uint right = (2 * i + 1 < n) ? input[2 * i + 1] : 0;
    levels[i] = left + right;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction_levels(
    __global uint* levels,
    unsigned int n,          // Hазмер уровня
    unsigned int in_offset,  // Где начинается уровень k-1
    unsigned int out_offset) // Где начинается уровень k
{
    const unsigned int i = get_global_id(0);
    if (i >= (n + 1) / 2)
        return;

    const uint left = levels[in_offset + 2 * i];
    const uint right = (2 * i + 1 < n) ? levels[in_offset + 2 * i + 1] : 0;
    levels[out_offset + i] = left + right;
}
