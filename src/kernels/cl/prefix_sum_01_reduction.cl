#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    __global uint* prefix_sum,
    unsigned int n,
    unsigned int offset,
    unsigned int sums_offset)
{
    __local uint sums[2 * GROUP_SIZE];
    const uint row = get_local_id(0);
    const size_t first = get_group_id(0) * (2 * GROUP_SIZE) + row;
    const size_t second = first + GROUP_SIZE;
    
    const uint a = first < n ? pow2_sum[offset + first] : 0;
    const uint b = second < n ? pow2_sum[offset + second] : 0;
    
    sums[row] = a;
    sums[row + GROUP_SIZE] = b;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 1; i < 2 * GROUP_SIZE; i *= 2) {
        const uint index = (row + 1) * 2 * i - 1;
        if (index < 2 * GROUP_SIZE) {
            sums[index] += sums[index - i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row == 0) {
        next_pow2_sum[sums_offset + get_group_id(0)] = sums[2 * GROUP_SIZE - 1];
        sums[2 * GROUP_SIZE - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = GROUP_SIZE; i > 0; i /= 2) {
        const uint index = (row + 1) * 2 * i - 1;
        if (index < 2 * GROUP_SIZE) {
            const uint left = sums[index - i];
            sums[index - i] = sums[index];
            sums[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (first < n) {
        prefix_sum[offset + first] = sums[row] + a;
    }
    if (second < n) {
        prefix_sum[offset + second] = sums[row + GROUP_SIZE] + b;
    }
}
