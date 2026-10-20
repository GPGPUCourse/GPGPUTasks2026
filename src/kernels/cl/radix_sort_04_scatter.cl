#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* counts,     // counts[bucket * num_groups + group]
    __global const uint* prefix_sum, // inclusive scan массива counts
    __global       uint* sorted_values,
    unsigned int n,
    unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int group = get_group_id(0);
    const unsigned int num_groups = get_num_groups(0);

    __local uint local_digits[GROUP_SIZE];

    const uint value = (gid < n) ? values[gid] : 0;
    // Нити за пределами n получают несуществующую корзину, чтобы не влиять на ранги.
    const uint digit = (gid < n) ? ((value >> bit_shift) & RADIX_MASK) : NUM_BUCKETS;
    local_digits[lid] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n)
        return;

    // Ранг внутри группы среди элементов с той же цифрой - сохраняет стабильность сортировки.
    uint rank = 0;
    for (unsigned int j = 0; j < lid; ++j)
        rank += (local_digits[j] == digit);

    const unsigned int idx = digit * num_groups + group;
    const uint offset = prefix_sum[idx] - counts[idx];
    sorted_values[offset + rank] = value;
}
