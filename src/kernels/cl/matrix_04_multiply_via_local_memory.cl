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
    __local float la[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float lb[GROUP_SIZE_Y][GROUP_SIZE_X];
    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    float cum = 0;

    for (size_t base = 0; base < k; base += GROUP_SIZE_X) {
        la[ly][lx] = (y < h && base + lx < k) ? a[y * k + base + lx] : 0;
        lb[ly][lx] = (x < w && base + ly < k) ? b[(base + ly) * w + x] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            cum += la[ly][i] * lb[i][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = cum;
    }
}
