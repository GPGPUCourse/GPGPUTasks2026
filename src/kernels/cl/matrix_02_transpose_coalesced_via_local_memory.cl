#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,
                       __global       float* transposed_matrix,
                                unsigned int w,
                                unsigned int h)
{
    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);
    const size_t bx = get_group_id(0) * get_local_size(0);
    const size_t by = get_group_id(1) * get_local_size(1);

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (bx + lx < w && by + ly < h) {
        tile[ly][lx] = matrix[(by + ly) * w + bx + lx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const size_t out_x = by + lx;
    const size_t out_y = bx + ly;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[lx][ly];
    }
}
