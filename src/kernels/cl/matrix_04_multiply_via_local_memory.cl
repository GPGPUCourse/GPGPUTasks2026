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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    float acc = 0.0f;
    for (unsigned int k0 = 0; k0 < k; k0 += GROUP_SIZE_X) {
        tile_a[local_j][local_i] = (j < h && k0 + local_i < k) ? a[j * k + k0 + local_i] : 0.0f;
        tile_b[local_j][local_i] = (i < w && k0 + local_j < k) ? b[(k0 + local_j) * w + i] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int t = 0; t < GROUP_SIZE_X; ++t)
            acc += tile_a[local_j][t] * tile_b[t][local_i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < w && j < h)
        c[j * w + i] = acc;
}
