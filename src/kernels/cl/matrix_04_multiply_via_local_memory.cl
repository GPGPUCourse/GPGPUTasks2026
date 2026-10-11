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
    // TODO
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    const unsigned int lcol = get_local_id(0);
    const unsigned int lrow = get_local_id(1);

    __local float localA[GROUP_SIZE_X][GROUP_SIZE_Y];
    __local float localB[GROUP_SIZE_X][GROUP_SIZE_Y];

    float cs = 0.0f;

    for (int q = 0; q < k; q += GROUP_SIZE_X) {
        if (row < h && q + lcol < k)
            localA[lrow][lcol] = a[row * k + q + lcol];
        else
            localA[lrow][lcol] = 0.0f;

        if (col < w && q + lrow < k)
            localB[lrow][lcol] = b[(q + lrow) * w + col];
        else
            localB[lrow][lcol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < GROUP_SIZE_X; ++p)
            cs += localA[lrow][p] * localB[p][lcol];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w)
        c[row * w + col] = cs;
}
