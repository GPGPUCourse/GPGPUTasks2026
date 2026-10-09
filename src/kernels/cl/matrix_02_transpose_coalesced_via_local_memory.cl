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
    __local float buffer[GROUP_SIZE_Y][GROUP_SIZE_X + 2];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    if (i < w && j < h) {
        buffer[local_j][local_i] = matrix[j * w + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    const unsigned int ti = (i - local_i) + local_j;
    const unsigned int tj = (j - local_j) + local_i;

    if (ti < w && tj < h) {
        transposed_matrix[ti * h + tj] = buffer[local_i][local_j];
    }
}
