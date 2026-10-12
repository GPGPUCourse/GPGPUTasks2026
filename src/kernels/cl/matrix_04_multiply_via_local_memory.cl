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
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);

    __local float cache_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float cache_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    for (size_t offset = 0; offset < k; offset += GROUP_SIZE_X) {
        if (y < h && offset + local_x < k) {
            cache_a[local_y][local_x] = a[y * k + offset + local_x];
        } else {
            cache_a[local_y][local_x] = 0;
        }

        if (offset + local_y < k && x < w) {
            cache_b[local_y][local_x] = b[(offset + local_y) * w + x];
        } else {
            cache_b[local_y][local_x] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < GROUP_SIZE_X; ++i) {
            sum += cache_a[local_y][i] * cache_b[i][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
