#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

//__attribute__((reqd_work_group_size(1, 16, 16)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float local_mem[16][16];

    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    const bool is_in_bound = global_x < w && global_y < h;

    if (is_in_bound) {
        local_mem[local_y][local_x] = matrix[global_y * w + global_x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (is_in_bound)
        transposed_matrix[global_x * h + global_y] = local_mem[local_y][local_x];
}
