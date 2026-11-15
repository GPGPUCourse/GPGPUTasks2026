#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void reduce_centroids(
    __global const float*  input_min,
    __global const float*  input_max,
    __global float*        output_min,
    __global float*        output_max,
    uint                   n
) {
    const uint thread_idx = get_global_id(0);
    const uint block_idx = get_local_id(0);
    const uint block_id = get_group_id(0);

    __local float min_x[GROUP_SIZE];
    __local float min_y[GROUP_SIZE];
    __local float min_z[GROUP_SIZE];

    __local float max_x[GROUP_SIZE];
    __local float max_y[GROUP_SIZE];
    __local float max_z[GROUP_SIZE];

    min_x[block_idx] = thread_idx < n ? input_min[thread_idx * 3 + 0] : +INFINITY;
    min_y[block_idx] = thread_idx < n ? input_min[thread_idx * 3 + 1] : +INFINITY;
    min_z[block_idx] = thread_idx < n ? input_min[thread_idx * 3 + 2] : +INFINITY;

    max_x[block_idx] = thread_idx < n ? input_max[thread_idx * 3 + 0] : -INFINITY;
    max_y[block_idx] = thread_idx < n ? input_max[thread_idx * 3 + 1] : -INFINITY;
    max_z[block_idx] = thread_idx < n ? input_max[thread_idx * 3 + 2] : -INFINITY;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
        if (block_idx < offset) {
            min_x[block_idx] = fmin(min_x[block_idx], min_x[block_idx + offset]);
            min_y[block_idx] = fmin(min_y[block_idx], min_y[block_idx + offset]);
            min_z[block_idx] = fmin(min_z[block_idx], min_z[block_idx + offset]);

            max_x[block_idx] = fmax(max_x[block_idx], max_x[block_idx + offset]);
            max_y[block_idx] = fmax(max_y[block_idx], max_y[block_idx + offset]);
            max_z[block_idx] = fmax(max_z[block_idx], max_z[block_idx + offset]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (block_idx == 0) {
        output_min[block_id * 3 + 0] = min_x[0];
        output_min[block_id * 3 + 1] = min_y[0];
        output_min[block_id * 3 + 2] = min_z[0];

        output_max[block_id * 3 + 0] = max_x[0];
        output_max[block_id * 3 + 1] = max_y[0];
        output_max[block_id * 3 + 2] = max_z[0];
    }
}