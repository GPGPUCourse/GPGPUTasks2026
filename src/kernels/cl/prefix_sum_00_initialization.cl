#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_00_initialization(
    __global const unsigned int* input, // contains n values
    __global       unsigned int* buffer_pow2_sum, // will contain n / (1 << pow2) values
    unsigned int min_precomputed_pow2,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local uint local_input[GROUP_SIZE];
    local_input[local_index] = index < n ? input[index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int window_size = 1 << min_precomputed_pow2;
    rassert(GROUP_SIZE % window_size == 0, 7996913263912);

    const unsigned int num_workers = GROUP_SIZE / window_size;

    if (local_index < num_workers) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < window_size; ++i) {
            sum += local_input[local_index * window_size + i];
        }
        const unsigned int output_index = get_group_id(0) * num_workers + local_index;
        buffer_pow2_sum[output_index] = sum;
    }
}
