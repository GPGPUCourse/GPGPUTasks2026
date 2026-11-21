#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // rows=h, cols=w
                       __global       float* transposed_matrix, // rows=w, cols=h
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);
    if (col >= w || row >= h)
        return;

    transposed_matrix[col * h + row] = matrix[row * w + col];
}
