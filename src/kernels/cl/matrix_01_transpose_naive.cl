#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (col >= w || row >= h)
        return;

    const int source_id = row * w + col;
    const float source_val = matrix[source_id];
    const int transpos_id = col * h + row;
    transposed_matrix[transpos_id] = source_val;
}
