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
    const unsigned int dx = get_local_id(0);
    const unsigned int dy = get_local_id(1);

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned base_x = x - dx;
    const unsigned base_y = y - dy;

    __local float chunk[GROUP_SIZE_X][GROUP_SIZE_Y + 1];
    const unsigned int local_index = dy * GROUP_SIZE_X + dx;
    const unsigned int global_index = y * w + x;

    if (x < w && y < h) {
        chunk[dy][dx] = matrix[global_index];
    } else {
        chunk[dy][dx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int new_dx = local_index % GROUP_SIZE_Y;
    const unsigned int new_dy = local_index / GROUP_SIZE_Y;
    const unsigned int target_x = base_y + new_dx;
    const unsigned int target_y = base_x + new_dy;
    if (target_x < h && target_y < w) {
        transposed_matrix[target_y * h + target_x] = chunk[new_dx][new_dy];
    }
}
