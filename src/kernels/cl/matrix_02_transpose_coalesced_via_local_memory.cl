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
    __local float local_matrix[(GROUP_SIZE_X + 1) * GROUP_SIZE_Y];

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    bool valid = x < w && y < h;

    if (valid) {
       local_matrix[ly * GROUP_SIZE_X + lx] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (!valid) return;

    const uint transposed_base = (x - lx) * h + (y - ly);
    // height is now width so y is multiplied by h
    const uint transposed_index = ly * h + lx;

    transposed_matrix[transposed_base + transposed_index] = local_matrix[lx * GROUP_SIZE_Y + ly];
}
