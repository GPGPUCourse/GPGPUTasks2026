#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pref_sum,
    __global const uint* arr_prev,
    __global const uint* block,
    __global       uint* arr_cur,
    unsigned int n,
    unsigned int bit)
{
    uint id = get_global_id(0);
    if (id >= n) return;
    uint gr = get_group_id(0);
    uint pref = pref_sum[id] + (gr == 0 ? 0 : block[gr - 1]);
    uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    if ((arr_prev[id] >> bit) & 1) {
        arr_cur[block[num_groups - 1] + id - pref] = arr_prev[id];
    } else {
        arr_cur[pref - 1] = arr_prev[id];
    }
}