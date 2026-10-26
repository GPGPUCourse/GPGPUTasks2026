#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#include "helpers/rassert.cl"
#include "../defines.h"

// Reduction of a flattened digit-major histogram.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* current_sums,
    __global uint* next_sums,
    uint current_n)
{
    const uint i = get_global_id(0);
    const uint left = 2u * i;
    if (left >= current_n) return;
    uint sum = current_sums[left];
    if (left + 1u < current_n) sum += current_sums[left + 1u];
    next_sums[i] = sum;
}
