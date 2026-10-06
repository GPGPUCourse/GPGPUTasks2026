#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    uint x = get_global_id(0), y = get_global_id(1);
    if (x < w && y < h)
    {
        float sum = 0.0;
        for (uint z = 0; z < k; ++z)
            sum += a[y * k + z] * b[z * w + x];
        c[y * w + x] = sum;
    }
}
