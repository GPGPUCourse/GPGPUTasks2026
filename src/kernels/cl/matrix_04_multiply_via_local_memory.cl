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
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);
    const uint col_local = get_local_id(0);
    const uint row_local = get_local_id(1);
    __local float tileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tileB[GROUP_SIZE_Y][GROUP_SIZE_X];
    float acc = 0;

    size_t iters = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (size_t i = 0; i < iters; i++) {

        uint col_a = i * GROUP_SIZE_X + col_local;
        if (col_a < k && row < h)
            tileA[row_local][col_local] = a[col_a + row * k];
        else
            tileA[row_local][col_local] = 0;

        uint row_b = i * GROUP_SIZE_Y + row_local;
        if (col < w && row_b < k)
            tileB[row_local][col_local] = b[row_b * w + col];
        else
            tileB[row_local][col_local] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t j = 0; j < GROUP_SIZE_X; j++) {
            acc += tileA[row_local][j] * tileB[j][col_local];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col < w && row < h)
        c[col + row * w] = acc;
}
