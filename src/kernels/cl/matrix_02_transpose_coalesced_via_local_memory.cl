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
    __local float buf[GROUP_SIZE_X][GROUP_SIZE_Y + 1];
    const unsigned int j = get_local_id(0);
    const unsigned int i = get_local_id(1);

    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    if (col < w && row < h)
        buf[i][j] = matrix[row * w + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int tcol = row - i + j;
    const unsigned int trow = col - j + i;

    if (trow < w && tcol < h)
        transposed_matrix[trow * h + tcol] = buf[j][i];
}
