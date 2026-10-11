#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_01_transpose_naive(
    __global const float* matrix, // width x height
    __global float* transposed_matrix, // height x width
    unsigned int width,
    unsigned int height)
{
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    if (col >= width || row >= height) {
        return;
    }

    transposed_matrix[col * height + row] = matrix[row * width + col];
}
