#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // Square 16x16 tile; padding reduces bank conflicts when reading columns.
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int x = get_group_id(0) * GROUP_SIZE_X + lx;
    const unsigned int y = get_group_id(1) * GROUP_SIZE_Y + ly;

    tile[ly][lx] = (x < w && y < h) ? matrix[(size_t)y * w + x] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Swap tile coordinates and local coordinates to write contiguous rows.
    const unsigned int tx = get_group_id(1) * GROUP_SIZE_Y + lx;
    const unsigned int ty = get_group_id(0) * GROUP_SIZE_X + ly;
    if (tx < h && ty < w)
        transposed_matrix[(size_t)ty * h + tx] = tile[lx][ly];
}
