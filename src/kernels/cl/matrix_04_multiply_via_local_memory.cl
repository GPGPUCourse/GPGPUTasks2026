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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    __local float tile_a[GROUP_SIZE_X][GROUP_SIZE_Y];
    __local float tile_b[GROUP_SIZE_X][GROUP_SIZE_Y];
    float sum = 0.0f;

    for (uint i = 0; i < k; i += GROUP_SIZE_X) {
        if (x < w && y < h) {
            tile_a[ly][lx] = a[y * k + i + lx];
            tile_b[ly][lx] = b[(i + ly) * w + x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (x < w && y < h) {
            for (uint j = 0; j < GROUP_SIZE_X; ++j) {
                sum += tile_a[ly][j] * tile_b[j][lx];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
