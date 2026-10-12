#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#define TILE 16

__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
        __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    const unsigned int row = get_group_id(1) * TILE + ly;
    const unsigned int col = get_group_id(0) * TILE + lx;

    float sum = 0.0f;

    const unsigned int num_tiles = (k + TILE - 1) / TILE;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        const unsigned int a_col = t * TILE + lx;
        const unsigned int b_row = t * TILE + ly;

        tileA[ly][lx] =
            (row < h && a_col < k) ? a[row * k + a_col] : 0.0f;

        tileB[ly][lx] =
            (b_row < k && col < w) ? b[b_row * w + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < TILE; ++i) {
            sum += tileA[ly][i] * tileB[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w) {
        c[row * w + col] = sum;
    }
}
