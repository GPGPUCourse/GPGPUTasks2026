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
    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);
    const uint input_i = get_group_id(0) * GROUP_SIZE_X + local_i;
    const uint input_j = get_group_id(1) * GROUP_SIZE_Y + local_j;

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (input_i < w && input_j < h) {
        tile[local_j][local_i] = matrix[input_j * w + input_i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint output_i = get_group_id(1) * GROUP_SIZE_Y + local_i;
    const uint output_j = get_group_id(0) * GROUP_SIZE_X + local_j;

    if (output_i < h && output_j < w) {
        transposed_matrix[output_j * h + output_i] = tile[local_i][local_j];
    }
}
