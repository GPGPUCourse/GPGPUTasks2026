#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1))) // intentionally twice X
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float buffer_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float buffer_b[GROUP_SIZE_X][GROUP_SIZE_X + 2];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int li = get_local_id(0);
    const unsigned int lj = get_local_id(1);

    float acc = 0.0f;

    for (unsigned int t = 0; t < k; t += GROUP_SIZE_X) {
        unsigned int ai = t + li;
        buffer_a[lj][li] = (j < h && ai < k) ? a[j * k + ai] : 0.0f;

        unsigned int bj = t + lj;
        buffer_b[lj][li] = (i < w && bj < k) ? b[bj * w + i] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int l = 0; l < GROUP_SIZE_X; l++) {
            acc += buffer_a[lj][l] * buffer_b[l][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < w && j < h) {
        c[j * w + i] = acc;
    }
}
