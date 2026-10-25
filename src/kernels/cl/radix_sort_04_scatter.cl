#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,      // n входных чисел
    __global const uint* group_bases, // group * 256 + digit: база группы для цифры
    __global       uint* out,         // n выходных чисел
    unsigned int n,
    unsigned int shift)               // номер младшего бита текущего 8-битного разряда
{
    const unsigned int group = get_global_id(0);
    const unsigned int group_start = group * GROUP_SIZE;
    const unsigned int group_size = min((unsigned)GROUP_SIZE, n - group_start);

    uint bases[256];
    for (unsigned int digit = 0; digit < 256; ++digit) {
        bases[digit] = group_bases[group * 256 + digit];
    }

    for (unsigned int i = 0; i < group_size; ++i) {
        const uint value = values[group_start + i];
        const uint digit = (value >> shift) & 255u;
        out[bases[digit]++] = value;
    }
}
