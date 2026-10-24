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
    __global       uint* count_for_group,
    unsigned int iter_num,
    unsigned int elems_per_group,
    unsigned int n)
{
    __local uint count[RADIX_NUM_MASK][GROUP_SIZE];

    const unsigned int local_ind = get_local_id(0);
    for (unsigned int mask = 0; mask < RADIX_NUM_MASK; mask++) {
        count[mask][local_ind] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int group_id = get_group_id(0);
    const unsigned int group_start = elems_per_group * group_id;
    const unsigned int group_end = group_start + elems_per_group < n ? group_start + elems_per_group : n;

    for (unsigned int i = group_start + local_ind; i < group_end; i += GROUP_SIZE) {
        const unsigned int part = (buffer1[i] >> (iter_num * RADIX_BITS)) & (RADIX_NUM_MASK - 1);
        count[part][local_ind]++;
    }

    for (unsigned int now_n = GROUP_SIZE / 2; now_n > 0; now_n /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_ind < now_n) {
            for (unsigned int mask = 0; mask < RADIX_NUM_MASK; mask++) {
                count[mask][local_ind] += count[mask][local_ind + now_n];
            }
        }
    }

    if (local_ind == 0) {
        const unsigned int num_groups = get_num_groups(0);
        for (unsigned mask = 0; mask < RADIX_NUM_MASK; mask++) {
            count_for_group[mask * num_groups + group_id] = count[mask][0];
        }
    }
}
