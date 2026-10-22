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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    const uint group_x = get_group_id(0);
    const uint group_y = get_group_id(1);

    __local float tile[16][16];

    if (x < w && y < h) {
        tile[ly][lx] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint out_x = group_y * 16 + lx;
    const uint out_y = group_x * 16 + ly;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[lx][ly];
    }
}
