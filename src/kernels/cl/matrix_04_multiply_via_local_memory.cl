#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#if GROUP_SIZE_X != GROUP_SIZE_Y
#error "Matrix multiplication requires a square work-group"
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);

    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const size_t group_size = get_local_size(0);
    const size_t steps_count = (k + group_size - 1) / group_size;

    float sum = 0.0f;
    for (size_t step = 0; step < steps_count; ++step) {
        const size_t a_col = step * group_size + lx;
        const size_t b_row = step * group_size + ly;

        if (y < h && a_col < k) {
            tile_a[ly][lx] = a[y * k + a_col];
        } else {
            tile_a[ly][lx] = 0.0f;
        }

        if (b_row < k && x < w) {
            tile_b[ly][lx] = b[b_row * w + x];
        } else {
            tile_b[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < group_size; ++i) {
            sum += tile_a[ly][i] * tile_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
