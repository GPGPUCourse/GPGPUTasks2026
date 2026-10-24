#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a,
    __global uint* b, // b[d * ng + g]
    unsigned int n,
    unsigned int bit)
{
    int i = get_global_id(0);
    int li = get_local_id(0);
    int g = get_group_id(0);
    int ng = get_num_groups(0);

    __local uint cnt[RADIX];

    if (li < RADIX) {
        cnt[li] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n) {
        int d = (a[i] >> bit) & (RADIX - 1);
        atomic_add(&cnt[d], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < RADIX) {
        b[li * ng + g] = cnt[li];
    }
}
