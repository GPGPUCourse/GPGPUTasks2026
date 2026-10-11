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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint x_local = get_local_id(0);
    const uint y_local = get_local_id(1);
    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (x < w && y < h)
        local_data[y_local][(x_local + y_local) % GROUP_SIZE_X] = matrix[x + y * w];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint x_transposed = get_group_id(1) * GROUP_SIZE_Y + x_local;
    const uint y_transposed = get_group_id(0) * GROUP_SIZE_X + y_local;

    if (x_transposed < h && y_transposed < w)
        transposed_matrix[x_transposed + y_transposed * h] = local_data[x_local][(x_local + y_local) % GROUP_SIZE_X];
}
