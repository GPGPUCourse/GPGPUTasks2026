#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* counts, // counts[bucket * num_groups + group] - сколько элементов группы попало в корзину
    unsigned int n,
    unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int group = get_group_id(0);
    const unsigned int num_groups = get_num_groups(0);

    __local uint local_counts[NUM_BUCKETS];
    if (lid < NUM_BUCKETS)
        local_counts[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n)
        atomic_inc(&local_counts[(values[gid] >> bit_shift) & RADIX_MASK]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Раскладка bucket-major: префиксная сумма по всему массиву counts
    // сразу дает глобальное смещение пары (bucket, group).
    if (lid < NUM_BUCKETS)
        counts[lid * num_groups + group] = local_counts[lid];
}
