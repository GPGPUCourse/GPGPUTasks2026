#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(TRANSPOSE_LOCAL_X, TRANSPOSE_LOCAL_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    const uint x = get_group_id(0) * TRANSPOSE_TILE_SIZE + local_x;
    const uint y = get_group_id(1) * TRANSPOSE_TILE_SIZE + local_y;

    __local float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];

    for (uint j = 0; j < TRANSPOSE_TILE_SIZE; j += TRANSPOSE_LOCAL_Y) {
        if (x < w && y + j < h) {
            tile[local_y + j][local_x] = matrix[(y + j) * w + x];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint out_x = get_group_id(1) * TRANSPOSE_TILE_SIZE + local_x;
    const uint out_y = get_group_id(0) * TRANSPOSE_TILE_SIZE + local_y;

    for (uint j = 0; j < TRANSPOSE_TILE_SIZE; j += TRANSPOSE_LOCAL_Y) {
        if (out_x < h && out_y + j < w) {
            transposed_matrix[(out_y + j) * h + out_x] = tile[local_x][local_y + j];
        }
    }
}
