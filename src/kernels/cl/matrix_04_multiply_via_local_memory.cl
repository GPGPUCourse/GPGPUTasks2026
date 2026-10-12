#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(MULTIPLY_TILE_SIZE, MULTIPLY_TILE_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    __local float tile_a[MULTIPLY_TILE_SIZE][MULTIPLY_TILE_SIZE];
    __local float tile_b[MULTIPLY_TILE_SIZE][MULTIPLY_TILE_SIZE];

    float acc = 0.0f;

    for (uint tile_k = 0; tile_k < k; tile_k += MULTIPLY_TILE_SIZE) {
        const uint a_x = tile_k + local_x;
        const uint b_y = tile_k + local_y;

        if (y < h && a_x < k) {
            tile_a[local_y][local_x] = a[y * k + a_x];
        } else {
            tile_a[local_y][local_x] = 0.0f;
        }

        if (b_y < k && x < w) {
            tile_b[local_y][local_x] = b[b_y * w + x];
        } else {
            tile_b[local_y][local_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint t = 0; t < MULTIPLY_TILE_SIZE; ++t) {
            acc += tile_a[local_y][t] * tile_b[t][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
