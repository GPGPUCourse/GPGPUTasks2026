#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_01_reduction(
    __global const uint* buf1,
    __global       uint* buf2,
    int level, // 1,2,4,8,16
    uint n) // total num elements in bufs
{
    int gid = get_global_id(0);
    int pair_element_id = gid - level;

    // check global id bound
    if (gid < 0 || gid >= n)
        return;

    uint buf1_val = buf1[gid];

    // check pair element id bound
    if (pair_element_id < 0)
    {
        buf2[gid] = buf1_val;
    }
    else
    {
        buf2[gid] = buf1_val + buf1[pair_element_id];
    }
}
