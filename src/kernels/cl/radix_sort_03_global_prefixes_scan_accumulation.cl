#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* accumulated,
    __global       uint* preffix,
    unsigned int n)
{
    const uint item_idx = get_global_id(0);
    const uint group_id = get_group_id(0);

    if (item_idx < n && group_id > 0) {
        preffix[item_idx] += accumulated[group_id - 1];
    }
}
