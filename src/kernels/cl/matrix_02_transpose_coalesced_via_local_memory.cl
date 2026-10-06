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
    __local float data[16][16];
    uint x = get_global_id(0), y = get_global_id(1);
    if (x < w && y < h) {
        uint x_local = get_local_id(0), y_local = get_local_id(1);
        data[x_local][y_local] = matrix[x + y * w];
        barrier(CLK_LOCAL_MEM_FENCE);
        transposed_matrix[(x - x_local) * h + (y - y_local) + h * y_local + x_local] = data[y_local][x_local];
    }
}
