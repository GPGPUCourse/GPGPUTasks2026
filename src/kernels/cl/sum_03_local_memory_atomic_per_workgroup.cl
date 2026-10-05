#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
    __global uint* sum,
    const unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint local_index = get_local_id(0);
    const size_t stride = get_global_size(0);
    uint partial = 0;

    for (size_t i = get_global_id(0); i < (size_t)n; i += stride)
        partial += a[i];

    __local uint local_data[GROUP_SIZE];
    local_data[local_index] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local_index < step)
            local_data[local_index] += local_data[local_index + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        atomic_add(sum, local_data[local_index]);
    }
}
