#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

//__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const int global_col = get_global_id(0);
    const int global_row = get_global_id(1);

    if (global_col >= (int)w || global_row >= (int)h)
        return;

    float total_sum = 0.0;

    for (int i = 0; i < k; ++i)
    {
        float a_val = a[global_row * k + i];
        float b_val = b[i * w + global_col];

        total_sum += a_val * b_val;
    }

    c[global_row * w + global_col] = total_sum;
}
