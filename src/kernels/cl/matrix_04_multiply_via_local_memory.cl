#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);

    float mult = 0.0f;
    for (size_t ofs = 0; ofs < k; ofs += GROUP_SIZE_X) {
        size_t a_col = ofs + lx;
        size_t b_row = ofs + ly;

        tile_a[ly][lx] = (y < h && a_col < k)
            ? a[y * k + a_col]
            : 0.0f;

        tile_b[ly][lx] = (x < w && b_row < k)
            ? b[b_row * w + x]
            : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < GROUP_SIZE_X; ++i) {
            mult += tile_a[ly][i] * tile_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = mult;
    }
}
