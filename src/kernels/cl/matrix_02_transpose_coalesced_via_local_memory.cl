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
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);

    tile[local_y][local_x] = (x < w && y < h) ? matrix[y * w + x] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    const size_t x2 = get_group_id(0) * GROUP_SIZE_X + local_y;
    const size_t y2 = get_group_id(1) * GROUP_SIZE_Y + local_x;

    if (y2 < h && x2 < w) {
        transposed_matrix[x2 * h + y2] = tile[local_x][local_y];
    }
}
