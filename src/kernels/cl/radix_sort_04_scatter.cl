#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,  // n входных чисел
    __global const uint* offsets, // buffer2[256 + g * 256 + d]: база группы g для цифры d
    __global       uint* out,     // n выходных чисел
    unsigned int n,
    unsigned int shift)
{
    const unsigned int group = get_global_id(0);
    const unsigned int group_start = group * GROUP_SIZE;
    const unsigned int group_n = min((unsigned)GROUP_SIZE, n - group_start);

    uint bases[256];
    for (unsigned int d = 0; d < 256; ++d) {
        bases[d] = offsets[256 + group * 256 + d];
    }

    for (unsigned int j = 0; j < group_n; ++j) {
        const uint value = values[group_start + j];
        const uint digit = (value >> shift) & 255u;
        out[bases[digit]++] = value;
    }
}
