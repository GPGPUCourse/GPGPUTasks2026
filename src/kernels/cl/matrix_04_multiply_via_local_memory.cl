#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float local_block_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float local_block_b[GROUP_SIZE_X][GROUP_SIZE_X];
    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);

    float acc = 0.0f;
    for (size_t offset = 0; offset < k; offset += GROUP_SIZE_X) {
        const size_t a_col = offset + local_x;
        const size_t b_row = offset + local_y;
        local_block_a[local_y][local_x] = (y < h && a_col < k) ? a[y * k + a_col] : 0.0f;
        local_block_b[local_y][local_x] = (b_row < k && x < w) ? b[b_row * w + x] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < GROUP_SIZE_X; ++i) {
            acc += local_block_a[local_y][i] * local_block_b[i][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
