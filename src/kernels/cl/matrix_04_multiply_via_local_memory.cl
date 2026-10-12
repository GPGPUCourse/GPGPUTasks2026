#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float lcl_a[16][16];
    __local float lcl_b[16][16];

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    float sum = 0.0f;

    for (unsigned int t = 0; t < k; t += 16) {
        if (y < h && (t + lx) < k) {
            lcl_a[ly][lx] = a[y * k + (t + lx)];
        } else {
            lcl_a[ly][lx] = 0.0f;
        }

        if (x < w && (t + ly) < k) {
            lcl_b[ly][lx] = b[(t + ly) * w + x];
        } else {
            lcl_b[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < 16; ++i) {
            sum += lcl_a[ly][i] * lcl_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
