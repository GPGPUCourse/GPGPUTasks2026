#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_03_multiply_naive(
    __global const float* a, // rows=height x cols=k
    __global const float* b, // rows=k x cols=width
    __global float* c, // rows=height x cols=width
    unsigned int width,
    unsigned int height,
    unsigned int k)
{
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    if (col >= width || row >= height) {
        return;
    }

    float cum_product = 0.0f;
    for (uint i = 0; i < k; ++i) {
        cum_product += a[row * k + i] * b[i * width + col];
    }

    c[row * width + col] = cum_product;
}
