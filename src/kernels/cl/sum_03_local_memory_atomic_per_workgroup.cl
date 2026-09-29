#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    local_data[local_index] = (index < n) ? a[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // тут прикольная идея, что все нити группы просто кладут в локальную память данные
    // а только одна нить делает по-настоящему суммирование и атомарно добавляет сумму
    // в глобальную память из локальной
    if (local_index == 0) {
        uint group_sum = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i)
            group_sum += local_data[i];
        atomic_add(sum, group_sum);
    }
}
