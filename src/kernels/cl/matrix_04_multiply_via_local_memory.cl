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
    const unsigned int tx = get_local_id(0);
    const unsigned int ty = get_local_id(1);
    const unsigned int col = get_group_id(0) * GROUP_SIZE_X + tx;
    const unsigned int row = get_group_id(1) * GROUP_SIZE_Y + ty;

    __local float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    float acc = 0.0f;
    const unsigned int tile = GROUP_SIZE_X;
    for (unsigned int t = 0; t < k; t += tile) {
        const unsigned int a_col = t + tx;
        const unsigned int b_row = t + ty;

        a_tile[ty][tx] = (row < h && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k && col < w) ? b[b_row * w + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        const unsigned int inner = (t + tile <= k) ? tile : (k - t);
        for (unsigned int ki = 0; ki < inner; ++ki)
            acc += a_tile[ty][ki] * b_tile[ki][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w)
        c[row * w + col] = acc;
}
