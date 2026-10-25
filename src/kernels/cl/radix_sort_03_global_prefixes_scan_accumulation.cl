#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* array_histograms,
    __global const uint* super_group_histograms,
    __global const uint* group_histograms,
    __global       uint* group_bases,
    unsigned int group_count)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int super_group = idx / 256;
    const unsigned int digit = idx % 256;

    // 1) старт: все элементы с цифрами < digit + элементы с цифрой digit в супер-группах < super_group
    uint running_offset = 0;
    for (unsigned int smaller_digit = 0; smaller_digit < digit; ++smaller_digit) {
        running_offset += array_histograms[smaller_digit];
    }
    for (unsigned int prev_super_group = 0; prev_super_group < super_group; ++prev_super_group) {
        running_offset += super_group_histograms[prev_super_group * 256 + digit];
    }

    // 2) идём по группам своей супер-группы, раздавая базы; последняя супер-группа может быть неполной
    const unsigned int group_end = min((super_group + 1) * 256, group_count);
    for (unsigned int group = super_group * 256; group < group_end; ++group) {
        group_bases[group * 256 + digit] = running_offset;
        running_offset += group_histograms[group * 256 + digit];
    }
}
