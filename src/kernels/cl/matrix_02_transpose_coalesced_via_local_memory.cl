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
    __local float buffer[(GROUP_SIZE_X+1)*GROUP_SIZE_Y];
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);

    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);

    if (global_x < w && global_y < h) {
        buffer[local_y * (GROUP_SIZE_X+1) + local_x] = matrix[global_y * w + global_x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    global_x = get_group_id(1) * GROUP_SIZE_Y + local_x;
    global_y = get_group_id(0) * GROUP_SIZE_X + local_y;
    if (global_x < h && global_y < w) {
        transposed_matrix[global_y * h + global_x] = buffer[local_x * (GROUP_SIZE_X+1) + local_y];
    }
}
