#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a,
    __global uint* b,
    unsigned int n)
{
    int j = get_global_id(0);
    int out_n = (n + REDUCE_FACTOR - 1) / REDUCE_FACTOR;
    if (j >= out_n)
        return;

    int base = j * REDUCE_FACTOR;
    uint s = 0;
    for (int t = 0; t < REDUCE_FACTOR; t++) {
        int i = base + t;
        if (i < n) {
            s += a[i];
        }
    }
    b[j] = s;
}
