#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* prefixes,
    __global const uint* group_prefixes,
    __global const uint* total,
    __global uint* output,
    unsigned int n,
    unsigned int bit)
{
    unsigned int gid = get_global_id(0);
    if (gid < n) {
        uint value = values[gid];
        uint zeros = group_prefixes[get_group_id(0)] + prefixes[gid];
        uint pos = ((value >> bit) & 1u) == 0 ? zeros - 1 : total[0] + gid - zeros;
        output[pos] = value;
    }
}
