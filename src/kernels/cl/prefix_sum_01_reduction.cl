#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* in_buf,
    __global       uint* out_buf,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int n)
{
    size_t id = get_global_id(0);
    if (id >= n) {
        return;
    }
    out_buf[out_offset + id] = in_buf[in_offset + 2 * id] + in_buf[in_offset + 2 * id + 1]; 
}
