#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#if SCAN_ITEMS_PER_WORK_ITEM < 1
#error SCAN_ITEMS_PER_WORK_ITEM must be positive
#endif

// NVIDIA OpenCL: обмен через регистры, с синхронизацией всех 32 потоков warp.
// Все потоки warp вызывают функцию, в том числе потоки с нулевыми значениями.
// width = 32 для элементов блока, width = 8 для сумм его восьми warp.
// (по сути своей замена subgroup операций, которые не поддерживаются на сервере)
static inline uint scan_warp_inclusive(uint value, uint width)
{
#pragma unroll
    for (uint offset = 1; offset < width; offset <<= 1) {
        __asm__ volatile(
            "{\n\t"
            " .reg .b32 neighbour;\n\t"
            " .reg .pred valid;\n\t"
            " shfl.sync.up.b32 neighbour|valid, %0, %1, 0, 0xffffffff;\n\t"
            " @valid add.u32 %0, %0, neighbour;\n\t"
            "}"
            : "+r"(value)
            : "r"(offset));
    }
    return value;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* prefix_sum,
    __global uint* block_sums,
    unsigned int n)
{
    const uint local_id = get_local_id(0);
    const uint warp_id = local_id / SCAN_WARP_SIZE;
    const uint lane = local_id % SCAN_WARP_SIZE;
    const size_t block_id = get_group_id(0);

    __local uint warp_offsets[SCAN_WARP_COUNT];

    const size_t first_index = block_id * SCAN_BLOCK_SIZE +
                              (size_t)local_id * SCAN_ITEMS_PER_WORK_ITEM;

    uint values[SCAN_ITEMS_PER_WORK_ITEM];
#if SCAN_ITEMS_PER_WORK_ITEM == 4
    if (first_index + SCAN_ITEMS_PER_WORK_ITEM <= n) {
        const uint4 loaded = vload4(0, input + first_index);
        values[0] = loaded.s0;
        values[1] = loaded.s1;
        values[2] = loaded.s2;
        values[3] = loaded.s3;
    } else
#endif
    {
#pragma unroll
        for (uint item = 0; item < SCAN_ITEMS_PER_WORK_ITEM; ++item) {
            const size_t index = first_index + item;
            values[item] = index < n ? input[index] : 0u;
        }
    }

    // Последовательный Scan своих элементов в регистрах. В общий Scan
    // передаётся только итоговая сумма каждого work-item.
    uint work_item_sum = 0;
#pragma unroll
    for (uint item = 0; item < SCAN_ITEMS_PER_WORK_ITEM; ++item) {
        work_item_sum += values[item];
        values[item] = work_item_sum;
    }
    const uint warp_prefix = scan_warp_inclusive(work_item_sum, SCAN_WARP_SIZE);

    if (lane == SCAN_WARP_SIZE - 1)
        warp_offsets[warp_id] = warp_prefix;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (warp_id == 0) {
        const uint warp_sum = lane < SCAN_WARP_COUNT ? warp_offsets[lane] : 0u;
        const uint sums_prefix = scan_warp_inclusive(warp_sum, SCAN_WARP_COUNT);

        if (lane < SCAN_WARP_COUNT)
            warp_offsets[lane] = sums_prefix - warp_sum;

        // Сумма всей WG станет элементом входа следующего уровня Scan.
        if (lane == SCAN_WARP_COUNT - 1)
            block_sums[block_id] = sums_prefix;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Здесь префиксы пока отсчитываются от начала своей WG.
    const uint work_item_offset = warp_offsets[warp_id] +
                                  warp_prefix - work_item_sum;
#pragma unroll
    for (uint item = 0; item < SCAN_ITEMS_PER_WORK_ITEM; ++item)
        values[item] += work_item_offset;

#if SCAN_ITEMS_PER_WORK_ITEM == 4
    if (first_index + SCAN_ITEMS_PER_WORK_ITEM <= n) {
        vstore4((uint4)(values[0], values[1], values[2], values[3]),
                0, prefix_sum + first_index);
    } else
#endif
    {
#pragma unroll
        for (uint item = 0; item < SCAN_ITEMS_PER_WORK_ITEM; ++item) {
            const size_t index = first_index + item;
            if (index < n)
                prefix_sum[index] = values[item];
        }
    }
}
