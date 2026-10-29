#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_simple_version1(
    __global const uint* a,
    __global       uint* b,
    unsigned int n,
    unsigned int pw2)
{
    int index = get_global_id(0);
    unsigned int actual_value = a[index];
    unsigned int prev_block = (index >= pw2 ? a[index - pw2] : 0);
    
    b[index] = actual_value + prev_block;
}
