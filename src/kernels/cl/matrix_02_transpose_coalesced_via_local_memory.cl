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
    // TODO
    __local float tile[TILE][TILE+1];
    const uint l_i = get_local_id(0);
    const uint l_j = get_local_id(1);

    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    if (i < w && j < h){
        tile[l_j][l_i] = matrix[j * w + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint out_x = get_group_id(1) * TILE + l_i;
    const uint out_y = get_group_id(0) * TILE + l_j;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[l_i][l_j];
    }

}
