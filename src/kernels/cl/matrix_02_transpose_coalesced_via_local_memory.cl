#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"
#define TILE 16

__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
     __local float tile[TILE][TILE];

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    const unsigned int col = get_group_id(0) * TILE + lx;
    const unsigned int row = get_group_id(1) * TILE + ly;

    if (col < w && row < h) {
        tile[ly][lx] = matrix[row * w + col];
    } else {
        tile[ly][lx] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int out_col = get_group_id(1) * TILE + lx;
    const unsigned int out_row = get_group_id(0) * TILE + ly;

    if (out_col < h && out_row < w) {
        transposed_matrix[out_row * h + out_col] = tile[lx][ly];
    }
}
