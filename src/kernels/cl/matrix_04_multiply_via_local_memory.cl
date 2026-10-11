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
    const unsigned int p = GROUP_SIZE_X;
    const unsigned int x = get_local_id(0), y = get_local_id(1);
    const unsigned int gx = get_global_id(0), gy = get_global_id(1);
    __local float la[p * p], lb[p * p];
    float acc = 0;
    for (unsigned int f = 0; f < k; f += p) {
        la[y * p + x] = (gy < h && f + x < k) ? a[gy * k + f + x] : 0.0f;
        lb[y * p + x] = (gx < w && f + y < k) ? b[(f + y) * w + gx] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int i = 0; i < p; ++i) {
            acc += la[y * p + i] * lb[i * p + x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gx < w && gy < h) {
        c[gy * w + gx] = acc;
    }
}
