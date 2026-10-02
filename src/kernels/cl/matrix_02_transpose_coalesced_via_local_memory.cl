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
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);
    size_t x = get_group_id(0) * GROUP_SIZE_X + lx;
    size_t y = get_group_id(1) * GROUP_SIZE_Y + ly;

    if (x < w && y < h) {
        tile[ly][lx] = matrix[y * w + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t output_x = get_group_id(1) * GROUP_SIZE_Y + lx;
    size_t output_y = get_group_id(0) * GROUP_SIZE_X + ly;
    if (output_x < h && output_y < w) {
        transposed_matrix[output_y * h + output_x] = tile[lx][ly];
    }
}
