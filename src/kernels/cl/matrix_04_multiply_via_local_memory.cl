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
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);
    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    const uint tiles_count = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (uint tile_index = 0; tile_index < tiles_count; ++tile_index) {
        const uint a_i = tile_index * GROUP_SIZE_X + local_i;
        const uint b_j = tile_index * GROUP_SIZE_Y + local_j;

        tile_a[local_j][local_i] =
            (j < h && a_i < k) ? a[j * k + a_i] : 0.0f;
        tile_b[local_j][local_i] =
            (b_j < k && i < w) ? b[b_j * w + i] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint ki = 0; ki < GROUP_SIZE_X; ++ki) {
            sum += tile_a[local_j][ki] * tile_b[ki][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < w && j < h) {
        c[j * w + i] = sum;
    }
}
