#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // rows=h, cols=w
                       __global       float* transposed_matrix, // rows=w, cols=h
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int tx = get_local_id(0);
    const unsigned int ty = get_local_id(1);
    const unsigned int col = get_group_id(0) * GROUP_SIZE_X + tx;
    const unsigned int row = get_group_id(1) * GROUP_SIZE_Y + ty;

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    tile[ty][tx] = (row < h && col < w) ? matrix[row * w + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int out_row = get_group_id(0) * GROUP_SIZE_X + ty;
    const unsigned int out_col = get_group_id(1) * GROUP_SIZE_Y + tx;
    if (out_row < w && out_col < h)
        transposed_matrix[out_row * h + out_col] = tile[tx][ty];
}
