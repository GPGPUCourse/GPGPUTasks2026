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


    __local float local_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    unsigned int i = get_global_id(0); //
    unsigned int j = get_global_id(1);
    unsigned int li = get_local_id(0);
    unsigned int lj = get_local_id(1);

    float acc = 0;

    for (int t = 0; t < k; t += GROUP_SIZE_X) {
        // загрузили a
        int a_col = t + li;
        if (j < h && a_col < k) {
            local_a[lj][li] = a[j * k + a_col];
        } else {
            local_a[lj][li] = 0;
        }

        // загрузили b
        int b_row = t + lj;
        if (i < w && b_row < k) {
            local_b[lj][li] = b[b_row * w + i];
        } else {
            local_b[lj][li] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // скалярно поумножали
        for (unsigned int p = 0; p < GROUP_SIZE_X; p++) {
            acc += local_a[lj][p] * local_b[p][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // записали
    if (i < w && j < h) {
        c[j * w + i] = acc;
    }
}
