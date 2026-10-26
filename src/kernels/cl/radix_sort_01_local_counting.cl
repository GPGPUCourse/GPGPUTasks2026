#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int radix_bits,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int t = get_local_id(0);
    const unsigned int j = get_global_id(0);
    const unsigned int gn = get_num_groups(0);
    const unsigned int g = get_group_id(0);
    const uint MASK = (1 << radix_bits);
    __local uint cnt[256];
    if (t < MASK) {
        cnt[t] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (j < n) {
        uint el = (buffer1[j] >> offset) & (MASK - 1);
        atomic_add(&cnt[el], 1u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < MASK) {
        buffer2[t * gn + g] = cnt[t];
    }
}
