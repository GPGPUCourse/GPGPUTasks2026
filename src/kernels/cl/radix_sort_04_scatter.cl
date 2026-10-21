#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* scan,
    __global const uint* total,
    __global uint* output,
    uint length,
    uint digit)
{
    const uint i = get_global_id(0);
    if (i >= length)
        return;

    const uint bit = (input[i] >> digit) & 1u;
    const uint prefix = scan[i];
    const uint destination = bit
        ? (length - total[0] + prefix)
        : (i - prefix);

    output[destination] = input[i];
}
