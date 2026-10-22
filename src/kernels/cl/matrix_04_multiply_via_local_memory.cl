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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    __local float tile_a[16][16];
    __local float tile_b[16][16];

    float acc = 0.0f;

    for (uint tile_index = 0; tile_index < (k / 16); ++tile_index) {
        tile_a[ly][lx] = a[y * k + tile_index * 16 + lx];
        tile_b[ly][lx] = b[(ly + tile_index * 16)* w + x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < 16; ++j) {
            acc += tile_a[ly][j] * tile_b[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
