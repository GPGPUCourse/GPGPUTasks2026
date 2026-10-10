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
    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);
    const size_t c_x = get_global_id(0);
    const size_t c_y = get_global_id(1);

    float acc = 0.0f;
    for (size_t k_offset = 0; k_offset < k; k_offset += GROUP_SIZE_X) {
        const size_t a_x = k_offset + local_x;
        const size_t b_y = k_offset + local_y;
        tile_a[local_y][local_x] = (c_y < h && a_x < k) ? a[c_y * k + a_x] : 0.0f;
        tile_b[local_y][local_x] = (b_y < k && c_x < w) ? b[b_y * w + c_x] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t tile_k = 0; tile_k < GROUP_SIZE_X; ++tile_k) {
            acc += tile_a[local_y][tile_k] * tile_b[tile_k][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (c_x < w && c_y < h) {
        c[c_y * w + c_x] = acc;
    }
}
