#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    volatile __global ulong* tile_state,
    volatile __global uint* next_tile,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index < n)
        tile_state[index] = 0ul;

    if (index == 0)
        next_tile[0] = 0;
}
