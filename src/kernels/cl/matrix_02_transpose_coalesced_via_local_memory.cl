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
    uint local_x = get_local_id(0);
    uint local_y = get_local_id(1);

    uint group_x = get_group_id(0);
    uint group_y = get_group_id(1);

    uint in_x = group_x * GROUP_SIZE_X + local_x;
    uint in_y = group_y * GROUP_SIZE_Y + local_y;

    if (in_x >= w || in_y >= h) {
        return;
    }

    buf[local_y][local_x] = matrix[in_y * w + in_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    uint out_x = group_y * GROUP_SIZE_X + local_x;
    uint out_y = group_x * GROUP_SIZE_Y + local_y;

    if (out_x >= h || out_y >= w) {
        return;
    }
    transposed_matrix[out_y * h + out_x] = buf[local_x][local_y];
}
