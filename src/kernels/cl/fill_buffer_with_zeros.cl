#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n_pow2, 
    unsigned int n)
{
    size_t id = get_global_id(0);
    
    if (id < n || id >= n_pow2) {
        return;
    }
    
    buffer[id] = 0;
}
