#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* values,
    __global uint* buffer1,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint scan[GROUP_SIZE];

    const uint valid = gid < n;
    const uint value = valid ? values[gid] : 0u;
    const uint digit = (value >> bit) & 1u;
    const uint is_zero = valid && digit == 0u;

    scan[lid] = is_zero;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const uint add = lid >= offset ? scan[lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        const uint zeros_before = scan[lid] - is_zero;
        const uint ones_before = lid - zeros_before;
        const uint rank = digit == 0u ? zeros_before : ones_before;

        uint encoded = (rank << 1) | digit;

        if (lid == GROUP_SIZE - 1 || gid == n - 1)
            encoded |= scan[lid] << 9;
        buffer1[gid] = encoded;
    }
}
