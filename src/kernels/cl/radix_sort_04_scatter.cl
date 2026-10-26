#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* numbers,
    __global const uint* local_pref_sums,
    __global const uint* group_pref_sums,
    __global       uint* output,
    uint bit,
    uint n)
{
    const uint item_idx = get_global_id(0);
    if (item_idx >= n) {
        return;
    }

    const uint group_id = get_group_id(0);
    const uint number = numbers[item_idx];
    const uint is_zero = ((number >> bit) & 1) == 0;
    const uint offset = group_id == 0 ? 0 : group_pref_sums[group_id - 1];

    const uint zeroes_before = offset + local_pref_sums[item_idx] - is_zero;
    if (is_zero) {
        output[zeroes_before] = number;
    } else {
        const uint groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
        const uint total_zeroes = group_pref_sums[groups - 1];
        output[total_zeroes + item_idx - zeroes_before] = number;
    }
}