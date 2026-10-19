#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* block_prefix_sum, // Глобальные inclusive-префиксы сумм блоков.
    __global uint* prefix_sum,            // Локальные префиксы; обновляем на месте.
    unsigned int n)
{
    const size_t block_id = get_group_id(0);
    // У первого блока смещение равно нулю, его результат уже готов.
    if (block_id == 0)
        return;

    const uint offset = block_prefix_sum[block_id - 1];
    const size_t first_index = block_id * SCAN_BLOCK_SIZE +
                              get_local_id(0) * SCAN_ITEMS_PER_WORK_ITEM;

#if SCAN_ITEMS_PER_WORK_ITEM == 4
    if (first_index + SCAN_ITEMS_PER_WORK_ITEM <= n) {
        const uint4 values = vload4(0, prefix_sum + first_index);
        vstore4(values + (uint4)(offset), 0, prefix_sum + first_index);
    } else
#endif
    {
#pragma unroll
        for (uint item = 0; item < SCAN_ITEMS_PER_WORK_ITEM; ++item) {
            const size_t index = first_index + item;
            if (index < n)
                prefix_sum[index] += offset;
        }
    }
}
