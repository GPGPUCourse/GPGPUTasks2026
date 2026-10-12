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
    float sum = 0.0f;
    for (size_t start = 0; start < k; start += GROUP_SIZE_X) {
        tile_a[ly][lx] = (y < h && start + lx < k)
            ? a[y * k + start + lx] : 0.0f;
        tile_b[ly][lx] = (x < w && start + ly < k)
            ? b[(start + ly) * w + x] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int t = 0; t < GROUP_SIZE_X; ++t) {
            sum += tile_a[ly][t] * tile_b[t][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
