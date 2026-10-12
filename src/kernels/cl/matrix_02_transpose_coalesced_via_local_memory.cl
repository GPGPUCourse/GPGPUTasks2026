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
    __local float lcl[16][17];

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned int x = get_group_id(0) * 16 + lx;
    unsigned int y = get_group_id(1) * 16 + ly;

    if (x < w && y < h) {
        lcl[ly][lx] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    x = get_group_id(1) * 16 + lx;
    y = get_group_id(0) * 16 + ly;

    if (x < h && y < w) {
        transposed_matrix[y * h + x] = lcl[lx][ly];
    }
}
