#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    size_t x = get_local_id(0);
    size_t y = get_local_id(1);
    size_t n = get_group_id(0);
    size_t m = get_group_id(1);
    size_t g_x = n * GROUP_SIZE_X + x;
    size_t g_y = m * GROUP_SIZE_Y + y;

    __local float tile[32 * 33];
    tile[x * 33 + y] = matrix[g_y * w + g_x];
    barrier(CLK_LOCAL_MEM_FENCE);
    transposed_matrix[(n * GROUP_SIZE_X + y) * h + (m * GROUP_SIZE_Y + x)] = tile[y * 33 + x];
}
