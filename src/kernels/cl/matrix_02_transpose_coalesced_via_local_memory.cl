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

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    __local float source[GROUP_SIZE_X][GROUP_SIZE_Y+1];

    if (x < w && y < h) {
        source[ly][lx] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint out_x = (y - ly) + lx;
    const uint out_y = (x - lx) + ly;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = source[lx][ly];
    }
}
