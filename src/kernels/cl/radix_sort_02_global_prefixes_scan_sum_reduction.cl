#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1)
{
    // TODO
    const unsigned int i = get_global_id(0);

    if (i >= (a1 + 1) / 2)
        return;

    uint s = buffer1[2 * i];
    if (2 * i + 1 < a1)
        s += buffer1[2 * i + 1];

    buffer2[i] = s;
}
