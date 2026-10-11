#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Пишу пояснения для себя, потому что пока не до конца быстро понимаю решения
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // n значений: суммы блоков размера 2^pow2
    __global       uint* next_pow2_sum, // (n+1)/2 значений: суммы блоков вдвое длиннее
    unsigned int n)
{
    // Нить i собирает пару (2i, 2i+1) в один элемент следующего уровня.
    // Пример: [1,2,3,4] → [1+2, 3+4] = [3,7]
    const unsigned int i = get_global_id(0);
    const unsigned int n_next = (n + 1u) / 2u;
    if (i >= n_next)
        return;

    const unsigned int left = pow2_sum[2u * i];
    // Если n нечётное, у последней пары правого элемента нет — считаем 0.
    const unsigned int right = (2u * i + 1u < n) ? pow2_sum[2u * i + 1u] : 0u;
    next_pow2_sum[i] = left + right;
}
