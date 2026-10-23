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
    int result_x = get_global_id(0);
    int result_y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    __local float local_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float local_b[GROUP_SIZE_X * GROUP_SIZE_Y];
    float result = 0;

    for (int i = 0; i < k; i += GROUP_SIZE_X) {
        local_a[local_y * GROUP_SIZE_X + local_x] = a[result_y * k + i + local_x];
        local_b[local_y * GROUP_SIZE_X + local_x] = b[(i + local_y) * w + result_x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < GROUP_SIZE_X; ++j) {
            result += local_a[local_y * GROUP_SIZE_X + j] * local_b[j * GROUP_SIZE_X + local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[result_y * w + result_x] = result;
}
