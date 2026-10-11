#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[j] = sum[j*2^pow2, (j+1)*2^pow2)
    __global       uint* prefix_sum_accum, // в итоге prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    // Хотим inclusive scan: prefix[i] = a[0] + ... + a[i].
    // Число (i+1) в двоичном виде говорит, какие блоки длины 2^pow2
    // полностью входят в отрезок [0, i].
    //
    // Пример a=[1,2,3,4], уровни:
    //   pow2=0: [1,2,3,4]
    //   pow2=1: [3,7]
    //   pow2=2: [10]
    // Для i=2 (нужно 1+2+3=6): (i+1)=3 = 0b11
    //   bit0 → +pow2_sum[2] (уровень 0) = 3
    //   bit1 → +pow2_sum[0] (уровень 1) = 3
    //   итого 6.
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    // Есть ли у i + 1 чисел в массиве: 0...i степень двойки pow2 входящая в сумму для i+1
    const unsigned int mask = 1u << pow2;
    if (((i + 1u) & mask) == 0u)
        return; // этот размер блока для позиции i не нужен

    // Индекс блока длины 2^pow2, который заканчивается ровно в i (включительно).
    const unsigned int block_idx = ((i + 1u) >> pow2) - 1u;
    prefix_sum_accum[i] += pow2_sum[block_idx];
}
