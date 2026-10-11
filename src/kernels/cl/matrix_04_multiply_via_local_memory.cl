#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    __local float tileA[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float tileB[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    float sum = 0.0f;
    for (uint t = 0; t < k; t += GROUP_SIZE_X) {
        const uint a_col = t + lx;
        const uint b_row = t + ly;
        tileA[ly][lx] = (y < h && a_col < k) ? a[y * k + a_col] : 0.0f;
        tileB[ly][lx] = (b_row < k && x < w) ? b[b_row * w + x] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            sum += tileA[ly][i] * tileB[i][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h)
        c[y * w + x] = sum;
}
