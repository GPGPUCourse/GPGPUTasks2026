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
    const uint j = get_global_id(0);
    const uint i = get_global_id(1);

    if (i >= h || j >= w) {
        return;
    }

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    __local float buff_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float buff_b[GROUP_SIZE_X][GROUP_SIZE_X];

    float sum = 0.0f;

    for (uint z = 0; z < k; z += GROUP_SIZE_X) {
        if (i < h && z + local_x < k) {
            buff_a[local_y][local_x] = a[i * k + z + local_x];
        }

        if (z + local_y < k && j < w) {
            buff_b[local_y][local_x] = b[(z + local_y) * w + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint q = 0; q < GROUP_SIZE_X; q++) {
            sum += buff_a[local_y][q] * buff_b[q][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[i * w + j] = sum;

}
