#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global uint* local_one_rank,
    __global uint* block_one_counts,
    uint n,
    uint bit_index)
{
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint g = get_group_id(0);

    const uint bit = (i < n)
        ? ((input[i] >> bit_index) & 1u)
        : 0u;

    __local uint shared_bits[GROUP_SIZE];

    shared_bits[lid] = bit;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        uint add = 0u;

        if (lid >= offset) {
            add = shared_bits[lid - offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        shared_bits[lid] += add;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n) {
        local_one_rank[i] = shared_bits[lid] - bit;
    }

    if (lid == GROUP_SIZE - 1) {
        block_one_counts[g] = shared_bits[lid];
    }
}