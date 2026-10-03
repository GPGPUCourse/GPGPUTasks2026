#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(TRANSPOSE_TILE_SIZE, TRANSPOSE_BLOCK_ROWS, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    size_t x = get_global_id(0);
    size_t y = get_group_id(1) * TRANSPOSE_TILE_SIZE + get_local_id(1);
    for (unsigned int row = 0; row < TRANSPOSE_TILE_SIZE; row += TRANSPOSE_BLOCK_ROWS) {
        if (x < w && y + row < h) {
            transposed_matrix[x * h + y + row] = matrix[(y + row) * w + x];
        }
    }
}
