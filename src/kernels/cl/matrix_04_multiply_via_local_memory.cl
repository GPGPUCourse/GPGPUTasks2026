#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float a_piece[16][17];
    __local float b_piece[16][17];
    uint x = get_global_id(0), y = get_global_id(1);
    if (x < w && y < h) {
        uint x_local = get_local_id(0), y_local = get_local_id(1);
        float sum = 0.0;
        for (uint z = 0; z * 16 < k; ++z) {
            a_piece[x_local][y_local] = a[y * k + z * 16 + x_local];
            b_piece[x_local][y_local] = b[(z * 16 + y_local) * w + x];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint i = 0; i < 16; ++i)
                sum += a_piece[i][y_local] * b_piece[x_local][i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        c[y * w + x] = sum;
    }
}
