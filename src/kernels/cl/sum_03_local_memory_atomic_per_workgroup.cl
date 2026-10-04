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
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    const bool is_not_masked = index < n / LOAD_K_VALUES_PER_ITEM;
    
    uint my_sum = 0;
    for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        if (is_not_masked) {
            my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
        }
    }

    local_data[local_index] = my_sum;

    uint shift = GROUP_SIZE / 2;
    uint left = local_index;
    while (shift > 0) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint right = local_index + shift;
        if (right < shift * 2) {
            local_data[left] += local_data[right]; 
        }
        shift /= 2;
    }

    if (local_index == 0) {
        atomic_add(sum, local_data[0]);
    }
}
