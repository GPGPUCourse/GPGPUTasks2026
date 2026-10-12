#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#if GROUP_SIZE_X != GROUP_SIZE_Y
#error "GROUP_SIZE_X and GROUP_SIZE_Y must be equal"
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    size_t local_col = get_local_id(0);
    size_t local_row = get_local_id(1);

    __local float local_a[GROUP_SIZE_Y * GROUP_SIZE_X];
    __local float local_b[GROUP_SIZE_Y * GROUP_SIZE_X];

    float sum = 0.0f;
    for (size_t i = 0; i < k; i += GROUP_SIZE_X) {
        local_a[local_row * GROUP_SIZE_X + (local_row + local_col) % GROUP_SIZE_X] =
            (i + local_col < k && row < h) ? a[row * k + (i + local_col)] : 0.0f;

        local_b[local_row * GROUP_SIZE_X + (local_row + local_col) % GROUP_SIZE_X] =
            (i + local_row < k && col < w) ? b[(i + local_row) * w + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t j = 0; j < GROUP_SIZE_X; ++j) {
            sum += local_a[local_row * GROUP_SIZE_X + (j + local_row) % GROUP_SIZE_X]
                * local_b[j * GROUP_SIZE_X + (j + local_col) % GROUP_SIZE_X];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col >= w || row >= h) return;

    c[row * w + col] = sum;
}
