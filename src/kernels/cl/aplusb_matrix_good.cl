#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

// Dimension 0 runs along a row, one uint4 per work-item.
// A warp of 32 work-items therefore reads 512 contiguous bytes in one instruction.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb_matrix_good(__global const uint* restrict a,
                     __global const uint* restrict b,
                     __global       uint* restrict c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int width4 = width >> 2;

    if (x >= width4 || y >= height)
        return;

    const unsigned int index4 = y * width4 + x;
    vstore4(vload4(index4, a) + vload4(index4, b), index4, c);
}
