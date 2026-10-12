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

    const unsigned int li = get_local_id(0);
    const unsigned int lj = get_local_id(1);

    const unsigned int tile_i0 = get_group_id(0) * GROUP_SIZE_X; 
    const unsigned int tile_j0 = get_group_id(1) * GROUP_SIZE_Y;

    const unsigned int i = tile_i0 + li;
    const unsigned int j = tile_j0 + lj;
    if (i < w && j < h) {
        tile[lj][li] = matrix[j * w + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int out_col = tile_j0 + li; 
    const unsigned int out_row = tile_i0 + lj;
    if (out_col < h && out_row < w) {
        transposed_matrix[out_row * h + out_col] = tile[li][lj];
    }
}
