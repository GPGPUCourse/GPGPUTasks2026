#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=height x cols=k
    __global const float* b, // rows=k x cols=width
    __global float* c, // rows=height x cols=width
    unsigned int width,
    unsigned int height,
    unsigned int k)
{
    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    __local float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    float cum_product = 0.0f;
    for (uint tile_start = 0; tile_start < k; tile_start += GROUP_SIZE_X) {

        if (row < height && tile_start + local_col < k) {
            a_tile[local_row][local_col] = a[row * k + tile_start + local_col];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        if (tile_start + local_row < k && col < width) {
            b_tile[local_row][local_col] = b[(tile_start + local_row) * width + col];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            cum_product += a_tile[local_row][i] * b_tile[i][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col < width && row < height) {
        c[row * width + col] = cum_product;
    }
}
