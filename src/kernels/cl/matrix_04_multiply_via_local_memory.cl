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
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int local_idx = local_x + local_y * GROUP_SIZE_X;

    __local float local_memory_a[GROUP_SIZE_X * GROUP_SIZE_Y]; // chunk of A
    __local float local_memory_b[GROUP_SIZE_X * GROUP_SIZE_Y]; // chunk of B

    float accum = 0;
    for (unsigned int k_shift = 0; k_shift < k; k_shift += GROUP_SIZE_X) {
        const unsigned int chunk_x_idx = k_shift + local_x;
        if (y < h && chunk_x_idx < k) {
            local_memory_a[local_idx] = a[chunk_x_idx + y * k];
        } else {
            local_memory_a[local_idx] = 0;
        }

        const unsigned int chunk_y_idx = k_shift + local_y;
        if (chunk_y_idx < k && x < w) {
            local_memory_b[local_idx] = b[x + chunk_y_idx * w];
        } else {
            local_memory_b[local_idx] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int ki = 0; ki < GROUP_SIZE_X; ++ki) {
            accum += local_memory_a[local_y * GROUP_SIZE_X + ki] * local_memory_b[ki * GROUP_SIZE_X + local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[x + y * w] = accum;
    }
}
