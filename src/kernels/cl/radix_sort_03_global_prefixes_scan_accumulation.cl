#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Для каждой цифры d вычисляет базу каждой группы: смещение группы g для цифры d =
// (количество всех элементов с цифрами < d) + (количество элементов с цифрой d в группах < g).
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* totals,   // buffer2[0..255]: суммарные количества цифр (результат кернела 02)
    __global const uint* histograms, // buffer1: локальные гистограммы групп, groups * 256 значений
    __global       uint* offsets,  // buffer2[256 + g * 256 + d]: база группы g для цифры d
    unsigned int groups)
{
    const unsigned int digit = get_global_id(0);

    uint running = 0;
    for (unsigned int d = 0; d < digit; ++d) {
        running += totals[d];
    }

    for (unsigned int group = 0; group < groups; ++group) {
        offsets[256 + group * 256 + digit] = running;
        running += histograms[group * 256 + digit];
    }
}
