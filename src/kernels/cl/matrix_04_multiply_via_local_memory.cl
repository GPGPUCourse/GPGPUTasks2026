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
    __local float green[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float blue[(GROUP_SIZE_X+1) * GROUP_SIZE_Y];
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);

    const size_t local_x = get_local_id(0);
    const size_t local_y = get_local_id(1);

    const size_t total_inbetween_sums = k / GROUP_SIZE_X + (k % GROUP_SIZE_X == 0 ? 0 : 1);
    for (size_t idx = 0; idx < total_inbetween_sums; idx++) {
        if (global_y < h && global_x < w) {
            green[local_y*GROUP_SIZE_X + local_x] = a[global_y*k + idx*GROUP_SIZE_X + local_x];
            blue[local_y*(GROUP_SIZE_X+1) + local_x] = b[(local_y + idx*GROUP_SIZE_Y)*w + global_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sum = 0;
        for (size_t sum_idx = 0; sum_idx < GROUP_SIZE_X; sum_idx++) {
            sum += green[local_y*GROUP_SIZE_X + sum_idx] * blue[sum_idx*(GROUP_SIZE_X+1) + local_x];
        }
        if (idx == 0) {
            c[global_y*w + global_x] = sum;
        } else {
            c[global_y*w + global_x] += sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}
