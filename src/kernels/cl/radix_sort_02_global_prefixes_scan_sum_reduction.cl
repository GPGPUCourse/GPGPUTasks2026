#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global uint* buffer2,
    unsigned int n)
{
    const uint group = get_global_id(0);
    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    if (group >= num_groups)
        return;

    const uint end = min((group + 1) * (uint)GROUP_SIZE, n) - 1;
    const uint zeros = buffer1[end] >> 9;
    const uint begin = group * GROUP_SIZE;
    const uint size = min((uint)GROUP_SIZE, n - begin);

    buffer2[group] = zeros;
    buffer2[num_groups + group] = size - zeros;
}