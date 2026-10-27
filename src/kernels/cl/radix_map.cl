#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_map(
    __global const uint* input,
    unsigned int iter,
    unsigned int mask,
    __global uint* bucket)
{
    unsigned int x = get_global_id(0);
    unsigned int marker = (input[x] >> (iter * RADIX_SHIFT)) & ((1 << RADIX_SHIFT) - 1);
    bucket[x] = (marker == mask ? 1 : 0);
}
