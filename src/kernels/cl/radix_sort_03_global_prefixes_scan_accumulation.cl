#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void
radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело оставляйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! ПЛЫВИТЕ ПО ТЕЧЕНИЮ!! ПОРЯДОК!!! ХАЙП!11!!1
    __global const uint* pow2_sum,
    __global const uint* input,
    __global uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    uint n,
    uint pow2,
    uint compression)
{
    uint id = get_global_id(0), sum1 = 0;
    uint x0 = (id >> 5 << 5 + compression) + id % 32;
    for (uint i = 5 + compression; i < pow2; ++i)
        if ((x0 + 1) & (1 << i))
            sum1 += pow2_sum[(1 << pow2 - i) + (x0 + 1 >> i) - 1];
    uint sum3 = 0;
    for (uint work_item = 0; work_item < (1 << compression); ++work_item) {
        uint x = x0 + 32 * work_item - 1;
        if (x < n) {
            uint sum2 = 0;
            for (uint i = 1; i < 5; ++i)
                if ((x + 1) & (1 << i))
                    sum2 += pow2_sum[(1 << pow2 - i) + (x + 1 >> i) - 1];
            if ((x + 1) & 1)
                sum2 += input[x];
            if (work_item != 0)
                for (uint i = 5; i < 5 + compression; ++i)
                    if ((x + 1 >> 5) % (1 << i - 5) == 0) {
                        if ((x + 1) & (1 << i))
                            sum3 += pow2_sum[(1 << pow2 - i) + (x + 1 >> i) - 1];
                        else
                            sum3 -= pow2_sum[(1 << pow2 - i) + (x + 1 >> i) - 2];
                    }
            prefix_sum_accum[x] = sum1 + sum2 + sum3;
        }
    }
}