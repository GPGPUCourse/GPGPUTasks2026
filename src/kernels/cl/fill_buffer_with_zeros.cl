#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n)
{
    const uint n4 = n >> 2;
    const uint i = get_global_id(0);

    __global uint4* buf4 = (__global uint4*)buffer;

    if (i < n4) {
        buf4[i] = (uint4)(0);
    } else if (i == n4) {
        for (uint k = 4 * n4; k < n; ++k)
            buffer[k] = 0;
    }
}