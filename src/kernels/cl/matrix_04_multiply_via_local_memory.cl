#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int x = get_global_id(0); 
    const unsigned int y = get_global_id(1); 

    float acc = 0.0f;
    for (unsigned int t0 = 0; t0 < k; t0 += TILE_SIZE) {
        const unsigned int a_col = t0 + lx;
        const unsigned int b_row = t0 + ly;
        tile_a[ly][lx] = (y < h && a_col < k) ? a[y * k + a_col] : 0.0f;
        tile_b[ly][lx] = (b_row < k && x < w) ? b[b_row * w + x] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int t = 0; t < TILE_SIZE; ++t) {
            acc += tile_a[ly][t] * tile_b[t][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
