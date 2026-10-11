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
    __local float submat[(GROUP_SIZE_X + 1) * GROUP_SIZE_Y];
    const unsigned int x = get_local_id(0), y = get_local_id(1);
    const unsigned int gx = get_global_id(0), gy = get_global_id(1);
    if (gx < w && gy < h) {
        submat[y * (GROUP_SIZE_X + 1) + x] = matrix[gy * w + gx];
    } else {
        submat[y * (GROUP_SIZE_X + 1) + x] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int out_row = get_group_id(0) * GROUP_SIZE_X + y;
    const unsigned int out_col = get_group_id(1) * GROUP_SIZE_Y + x;
    if (out_row < w && out_col < h)
        transposed_matrix[out_row * h + out_col] = submat[x * (GROUP_SIZE_X + 1) + y];
}
