#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const uint* matrix,            // w x h
                       __global       uint* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local uint data[GROUP_SIZE_X][GROUP_SIZE_Y];
    uint x = get_global_id(0), y = get_global_id(1);
    uint x_local = get_local_id(0), y_local =get_local_id(1);
    if (x < w && y < h) {
        data[x_local][y_local] = matrix[x + y * w];
    }

        barrier(CLK_LOCAL_MEM_FENCE);
    if (x-x_local+y_local<w && y-y_local+x_local<h) {
        transposed_matrix[(x - x_local) * h + (y - y_local) + h * y_local + x_local] = data[y_local][x_local];
    }
}
