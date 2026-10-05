#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    // TODO
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);
    size_t groups = get_num_groups(0);
    size_t step = get_global_size(0);

    uint thread_sum = 0;

    for (size_t i = index; i < n; i += step) {
        thread_sum += a[i];
    }
    __local uint local_data[GROUP_SIZE];
    local_data[local_index] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index==0){
        uint group_sum = 0;
        for (uint idx = 0; idx < GROUP_SIZE; idx++){
            group_sum += local_data[idx];
        }
        b[group_index] = group_sum;
    }
}
