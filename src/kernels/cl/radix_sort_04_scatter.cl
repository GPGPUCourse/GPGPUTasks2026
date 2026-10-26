#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global       uint* buffer3,
    unsigned int n,
    unsigned int l)
{
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    const uint key = buffer1[i];
    const uint b = (key >> l) & 1;
    const uint ones_before = buffer2[i] - b;
    const uint total_zeros = n - buffer2[n - 1];

    const uint pos = b ? (total_zeros + ones_before) : (i - ones_before);
    buffer3[pos] = key;
}