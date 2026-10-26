#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* local_one_rank,
    __global const uint* block_one_counts,
    __global const uint* global_one_prefix,
    __global uint* output,
    uint n,
    uint num_groups,
    uint bit_index)
{
    const uint i = get_global_id(0);

    if (i >= n) {
        return;
    }

    const uint g = get_group_id(0);

    const uint value = input[i];
    const uint bit = (value >> bit_index) & 1u;

    const uint local_ones = local_one_rank[i];
    const uint global_ones = global_one_prefix[g];

    const uint last_group = num_groups - 1u;

    const uint total_ones =
        global_one_prefix[last_group] +
        block_one_counts[last_group];

    const uint total_zeros = n - total_ones;

    uint pos;

    if (bit == 0u) {
        pos = i - global_ones - local_ones;
    } else {
        pos = total_zeros + global_ones + local_ones;
    }

    output[pos] = value;
}