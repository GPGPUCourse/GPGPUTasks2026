#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_03_scatter_kv(
    __global const uint* keys_in,
    __global const uint* values_in,
    __global const uint* scan,
    __global const uint* total,
    __global uint* keys_out,
    __global uint* values_out,
    uint length,
    uint digit)
{
    const uint i = get_global_id(0);
    if (i >= length)
        return;

    const uint key = keys_in[i];
    const uint bit = (key >> digit) & 1u;
    const uint prefix = scan[i];
    const uint destination = bit
        ? (length - total[0] + prefix)
        : (i - prefix);

    keys_out[destination] = key;
    values_out[destination] = values_in[i];
}