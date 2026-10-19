#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void decompress(
    __global uint* pref_sum,
    __global const uint* block,
    unsigned int n)
{
    const uint g = get_global_id(0);
    const uint gr = get_group_id(0);
    if (g >= n) {
        return;
    }
    if (gr > 0) {
        pref_sum[g] += block[gr - 1];
    }
}
