#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#if GROUP_SIZE_X != GROUP_SIZE_Y
#error These tiled kernels require square work groups
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);
    const size_t x = get_group_id(0) * GROUP_SIZE_X + lx;
    const size_t y = get_group_id(1) * GROUP_SIZE_Y + ly;
    tile[ly][lx] = (x < w && y < h) ? matrix[y * w + x] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    const size_t out_x = get_group_id(1) * GROUP_SIZE_Y + lx;
    const size_t out_y = get_group_id(0) * GROUP_SIZE_X + ly;
    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[lx][ly];
    }
}
