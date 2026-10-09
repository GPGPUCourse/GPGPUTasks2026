#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float local_matrix[256 + 16];

    if (x < w && y < h) {
        local_matrix[local_y * 16 + local_x] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int transposed_x = get_group_id(1) * 16 + local_x;
    const unsigned int transposed_y = get_group_id(0) * 16 + local_y;

    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = local_matrix[local_x * 16 + local_y];
    }
}
