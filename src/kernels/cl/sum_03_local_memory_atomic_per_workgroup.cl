#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    // Сначала собираем кусок массива в локальную память группы.
    // Дерево считает сумму группы, и в глобальный аккумулятор пишет
    // только мастер (local id 0): один atomicAdd на work-group.
    __local uint local_data[GROUP_SIZE];
    local_data[local_index] = (index < n) ? a[index] : 0u;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE / 2; offset > 0u; offset >>= 1) {
        if (local_index < offset) {
            local_data[local_index] += local_data[local_index + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0u) {
        atomic_add(sum, local_data[0]);
    }
}
