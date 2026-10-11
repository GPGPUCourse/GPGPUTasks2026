#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // width x height
    __global float* transposed_matrix, // height x width
    unsigned int width,
    unsigned int height)
{
    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint input_col = get_group_id(0) * GROUP_SIZE_X + local_col;
    const uint input_row = get_group_id(1) * GROUP_SIZE_Y + local_row;

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (input_col < width && input_row < height) {
        tile[local_row][local_col] = matrix[input_row * width + input_col];
    } else {
        tile[local_row][local_col] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint output_col = get_group_id(1) * GROUP_SIZE_Y + local_col;
    const uint output_row = get_group_id(0) * GROUP_SIZE_X + local_row;

    if (output_col < height && output_row < width) {
        transposed_matrix[output_row * height + output_col] = tile[local_col][local_row];
    }
}
