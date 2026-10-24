#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* histograms,  // локальные гистограммы всех групп: groups * 256 значений
    __global       uint* totals,      // суммарное количество каждой из 256 цифр
    unsigned int groups)
{
    const unsigned int digit = get_global_id(0);

    uint sum = 0;
    for (unsigned int group = 0; group < groups; ++group) {
        sum += histograms[group * 256 + digit];
    }
    totals[digit] = sum;
}
