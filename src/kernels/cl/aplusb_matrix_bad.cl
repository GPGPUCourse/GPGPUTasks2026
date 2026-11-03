#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

// Dimension 0 runs down a column. Neighboring work-items touch addresses
// `width` uints apart (64 KiB on the assignment matrix), so a warp cannot merge them.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int y = get_global_id(0);
    const unsigned int x = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const unsigned int index = y * width + x;
    c[index] = a[index] + b[index];
}
