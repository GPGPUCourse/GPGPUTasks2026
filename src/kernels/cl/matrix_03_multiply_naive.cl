#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const size_t c_x = get_global_id(0);
    const size_t c_y = get_global_id(1);

    if (c_x >= w || c_y >= h)
        return;

    float acc = 0.0f;
    for (size_t k_index = 0; k_index < k; ++k_index) {
        acc += a[c_y * k + k_index] * b[k_index * w + c_x];
    }

    c[c_y * w + c_x] = acc;
}
