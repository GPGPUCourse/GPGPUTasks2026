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
    __local float local_data[(GROUP_SIZE_X + 1) * GROUP_SIZE_Y];
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int base_x = global_x - local_x;
    int base_y = global_y - local_y;

    local_data[local_y * (GROUP_SIZE_X + 1) + local_x] = matrix[global_y * w + global_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed_matrix[(base_x + local_y) * h + base_y + local_x] = local_data[local_x * (GROUP_SIZE_X + 1) + local_y];
}
