#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

// Тут (и в других kernel'ах) еще пробовал добавить restrict, но на моей машине это не дало ничего
__attribute__((reqd_work_group_size(TRANSPOSE_TILE_SIZE, TRANSPOSE_BLOCK_ROWS, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // В теории, тут могут быть bank-конфликты
    // На практике, на моей машине такая версия и версия с +1 не имеют заметных отличий по производительности
    __local float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE];
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);
    size_t x = get_group_id(0) * TRANSPOSE_TILE_SIZE + lx;
    size_t y = get_group_id(1) * TRANSPOSE_TILE_SIZE + ly;

    for (unsigned int row = 0; row < TRANSPOSE_TILE_SIZE; row += TRANSPOSE_BLOCK_ROWS) {
        if (x < w && y + row < h) {
            tile[ly + row][lx] = matrix[(y + row) * w + x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t output_x = get_group_id(1) * TRANSPOSE_TILE_SIZE + lx;
    size_t output_y = get_group_id(0) * TRANSPOSE_TILE_SIZE + ly;
    for (unsigned int row = 0; row < TRANSPOSE_TILE_SIZE; row += TRANSPOSE_BLOCK_ROWS) {
        if (output_x < h && output_y + row < w) {
            transposed_matrix[(output_y + row) * h + output_x] = tile[lx][ly + row];
        }
    }
}
