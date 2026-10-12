#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float a_local[BLOCK_SIZE * BLOCK_SIZE];
    __local float b_local[BLOCK_SIZE * BLOCK_SIZE];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    float sum = 0.0f;
    for (int offset = 0; offset < k; offset += BLOCK_SIZE) {
        if (x < w && y < h) {
            a_local[ly * BLOCK_SIZE + lx] = a[y * k + (offset + lx)];
            b_local[ly * BLOCK_SIZE + lx] = b[(offset + ly) * w + x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (x < w && y < h) {
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                sum += a_local[ly * BLOCK_SIZE + i] * b_local[i * BLOCK_SIZE + lx];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = sum;
}
