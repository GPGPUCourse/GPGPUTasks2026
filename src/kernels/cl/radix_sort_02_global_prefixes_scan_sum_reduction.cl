#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* values,
    __global uint* block_sums,
    unsigned int n)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);

    __local uint prefixes[2][GROUP_SIZE];
    __local uint* current = prefixes[0];
    __local uint* next = prefixes[1];

    current[local_id] = global_id < n ? values[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        next[local_id] = current[local_id] + (local_id >= offset ? current[local_id - offset] : 0);

        barrier(CLK_LOCAL_MEM_FENCE);

        __local uint* temp = current;
        current = next;
        next = temp;
    }

    if (global_id < n) {
        values[global_id] = current[local_id];
    }

    if (local_id == GROUP_SIZE - 1) {
        block_sums[get_group_id(0)] = current[local_id];
    }
}
