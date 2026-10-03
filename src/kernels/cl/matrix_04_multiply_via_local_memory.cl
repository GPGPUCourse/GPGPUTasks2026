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
    __local float tile_a[GROUP_SIZE_Y * MULTIPLY_OUTPUTS_PER_THREAD][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X * MULTIPLY_OUTPUTS_PER_THREAD];
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);
    size_t x = get_group_id(0) * GROUP_SIZE_X * MULTIPLY_OUTPUTS_PER_THREAD + lx;
    size_t y = get_group_id(1) * GROUP_SIZE_Y * MULTIPLY_OUTPUTS_PER_THREAD + ly;
    // Каждый поток считает четыре элемента блока C размером 32x32.
    float sum00 = 0.0f, sum01 = 0.0f;
    float sum10 = 0.0f, sum11 = 0.0f;

    for (size_t offset = 0; offset < k; offset += GROUP_SIZE_X) {
        tile_a[ly][lx] = (y < h && offset + lx < k)
            ? a[y * k + offset + lx] : 0.0f;
        tile_a[ly + GROUP_SIZE_Y][lx] = (y + GROUP_SIZE_Y < h && offset + lx < k)
            ? a[(y + GROUP_SIZE_Y) * k + offset + lx] : 0.0f;
        tile_b[ly][lx] = (offset + ly < k && x < w)
            ? b[(offset + ly) * w + x] : 0.0f;
        tile_b[ly][lx + GROUP_SIZE_X] = (offset + ly < k && x + GROUP_SIZE_X < w)
            ? b[(offset + ly) * w + x + GROUP_SIZE_X] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            float a0 = tile_a[ly][i];
            float a1 = tile_a[ly + GROUP_SIZE_Y][i];
            float b0 = tile_b[i][lx];
            float b1 = tile_b[i][lx + GROUP_SIZE_X];
            sum00 = fma(a0, b0, sum00);
            sum01 = fma(a0, b1, sum01);
            sum10 = fma(a1, b0, sum10);
            sum11 = fma(a1, b1, sum11);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (y < h) {
        if (x < w) c[y * w + x] = sum00;
        if (x + GROUP_SIZE_X < w) c[y * w + x + GROUP_SIZE_X] = sum01;
    }
    if (y + GROUP_SIZE_Y < h) {
        if (x < w) c[(y + GROUP_SIZE_Y) * w + x] = sum10;
        if (x + GROUP_SIZE_X < w) c[(y + GROUP_SIZE_Y) * w + x + GROUP_SIZE_X] = sum11;
    }
}
