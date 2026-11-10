#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 32, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    int loc_i = get_local_id(1);
    int loc_j = get_local_id(0);

    int i = get_global_id(1);
    int j = get_global_id(0);

    __local float tmp_a[32][32];
    __local float tmp_b[32][32];

    float acc = 0;

    for (int base = 0; base < k; base += 32) {
        tmp_a[loc_i][loc_j] = i < h && base + loc_j < k ? a[i * k + base + loc_j] : 0;
        tmp_b[loc_i][loc_j] = j < w && base + loc_i < k ? b[(base + loc_i) * w + j] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < 32; ++t) {
            acc += tmp_a[loc_i][t] * tmp_b[t][loc_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < h && j < w) {
        c[i * w + j] = acc;
    }
}
