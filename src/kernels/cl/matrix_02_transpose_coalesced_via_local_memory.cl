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

    __local float a[16][17];

    size_t j = get_global_id(0);
    size_t i = get_global_id(1);

    size_t loc_j = get_local_id(0);
    size_t loc_i = get_local_id(1);

    if (i < h && j < w) {
        a[loc_j][loc_i] = matrix[i * w + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    size_t out_i = get_group_id(0) * 16 + loc_i;
    size_t out_j = get_group_id(1) * 16 + loc_j;
    
    if (out_i < w && out_j < h) {
        transposed_matrix[out_i * h + out_j] = a[loc_i][loc_j];
    }
}
