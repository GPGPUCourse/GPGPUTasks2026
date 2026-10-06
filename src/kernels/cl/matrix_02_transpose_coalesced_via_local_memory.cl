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
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int src_idx = x + y * w;
    if (x >= w || y >= h) {
        return;
    }

    const unsigned int local_stride = GROUP_SIZE_X + 1;

    const unsigned int warp_x = get_local_id(0);
    const unsigned int warp_y = get_local_id(1);
    const unsigned int local_write_idx = warp_x + warp_y * local_stride;
    const unsigned int local_read_idx = warp_y + warp_x * local_stride;

    __local float local_data[GROUP_SIZE_Y * local_stride];
    if (src_idx < w * h) {
        local_data[local_write_idx] = matrix[src_idx];
    } else {
        local_data[local_write_idx] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);

    const unsigned int out_x = group_y * GROUP_SIZE_X + warp_x;
    const unsigned int out_y = group_x * GROUP_SIZE_Y + warp_y;
    const unsigned int dst_idx = out_x + out_y * h;

    transposed_matrix[dst_idx] = local_data[local_read_idx];
}
