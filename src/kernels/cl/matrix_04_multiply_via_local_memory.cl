#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(MUL_GROUP_SIZE_X, MUL_GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    size_t x = get_local_id(0);
    size_t y = get_local_id(1);
    size_t g_x = get_group_id(0) * (THREAD_TILE_SIZE * MUL_GROUP_SIZE_X) + x;
    size_t g_y = get_group_id(1) * (THREAD_TILE_SIZE * MUL_GROUP_SIZE_Y) + y;

    __local float tile_a[THREAD_TILE_SIZE * MUL_GROUP_SIZE_Y][MUL_GROUP_SIZE_X];
    __local float tile_b[MUL_GROUP_SIZE_X][THREAD_TILE_SIZE * MUL_GROUP_SIZE_X];
    float acc[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {};

    for (size_t k_base = 0; k_base < k; k_base += MUL_GROUP_SIZE_X) {
        for (size_t ry = 0; ry < THREAD_TILE_SIZE; ++ry) {
            size_t local_row = y + ry * MUL_GROUP_SIZE_Y;
            size_t a_row = g_y + ry * MUL_GROUP_SIZE_Y;
            size_t a_col = k_base + x;
            tile_a[local_row][x] = (a_row < h && a_col < k)
                ? a[a_row * k + a_col] : 0.0f;
        }

        for (size_t tile_row = y; tile_row < MUL_GROUP_SIZE_X; tile_row += MUL_GROUP_SIZE_Y) {
            for (size_t rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                size_t local_col = x + rx * MUL_GROUP_SIZE_X;
                size_t b_row = k_base + tile_row;
                size_t b_col = g_x + rx * MUL_GROUP_SIZE_X;
                tile_b[tile_row][local_col] = (b_row < k && b_col < w)
                    ? b[b_row * w + b_col] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < MUL_GROUP_SIZE_X; ++i) {
            for (size_t ry = 0; ry < THREAD_TILE_SIZE; ++ry) {
                float a_value = tile_a[y + ry * MUL_GROUP_SIZE_Y][i];
                for (size_t rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
                    acc[ry][rx] += a_value * tile_b[i][x + rx * MUL_GROUP_SIZE_X];
                }
            }
        }

        // Finish reading the current tiles before loading the next pair.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t ry = 0; ry < THREAD_TILE_SIZE; ++ry) {
        size_t out_y = g_y + ry * MUL_GROUP_SIZE_Y;
        for (size_t rx = 0; rx < THREAD_TILE_SIZE; ++rx) {
            size_t out_x = g_x + rx * MUL_GROUP_SIZE_X;
            if (out_x < w && out_y < h) {
                c[out_y * w + out_x] = acc[ry][rx];
            }
        }
    }
}
