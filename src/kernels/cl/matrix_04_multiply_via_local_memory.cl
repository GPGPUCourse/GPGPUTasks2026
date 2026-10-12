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
    __local float a_tile[TILE][TILE];
    __local float b_tile[TILE][TILE];

    const uint l_x = get_local_id(0);
    const uint l_y = get_local_id(1);

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    float sum = 0.0f;
    for (uint t = 0; t < k; t += TILE) {
        if (y < h && t + l_x < k) a_tile[l_y][l_x] = a[y * k + t + l_x];
        else a_tile[l_y][l_x] = 0.0f;
        if (t + l_y < k && x < w) b_tile[l_y][l_x] = b[(t + l_y) * w + x];
        else b_tile[l_y][l_x] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < TILE; ++i)
            sum += a_tile[l_y][i] * b_tile[i][l_x];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h)
        c[y * w + x] = sum;
}
