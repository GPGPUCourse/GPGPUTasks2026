#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE 16

__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[TILE][TILE];
    __local float tile_b[TILE][TILE];

    const int global_x = get_global_id(0);   // column of C
    const int global_y = get_global_id(1);   // row of C
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    const int num_tiles = (k + TILE - 1) / TILE;
    float total_sum = 0.0;

    for (int t = 0; t < num_tiles; ++t)
    {
        const int col_a = t * TILE + local_x;
        const int row_b = t * TILE + local_y;

        tile_a[local_y][local_x] = (global_y < h && col_a < k) ? a[global_y * k + col_a] : 0.0;
        tile_b[local_y][local_x] = (row_b < k && global_x < w) ? b[row_b * w + global_x] : 0.0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE; ++i)
            total_sum += tile_a[local_y][i] * tile_b[i][local_x];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_y < h && global_x < w)
        c[global_y * w + global_x] = total_sum;
}
