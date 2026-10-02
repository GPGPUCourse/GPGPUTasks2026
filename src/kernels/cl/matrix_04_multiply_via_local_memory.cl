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
    __local float chunk_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float chunk_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int dx = get_local_id(0);
    const unsigned int dy = get_local_id(1);

    float accum = 0;

    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        const unsigned int global_index_a = y * k + i + dx;
        chunk_a[dy][dx] = (y < h && i + dx < k) ? a[global_index_a] : 0;

        const unsigned int global_index_b = (dy + i) * w + x;
        chunk_b[dy][dx] = (dy + i < k && x < w) ? b[global_index_b] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; j++) {
            accum += chunk_a[dy][j] * chunk_b[j][dx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const unsigned global_index_c = y * w + x;
    if (y < h && x < w) {
        c[global_index_c] = accum;
    }
}
