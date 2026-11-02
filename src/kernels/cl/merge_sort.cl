#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const size_t i = get_global_id(0);
    if (i >= (uint)n)
        return;

    // На входе отсортированы участки длины sorted_k. Сливаем соседние пары.
    const uint run_size = (uint)sorted_k;
    const uint pair_start = ((uint)i / (2 * run_size)) * (2 * run_size);
    const uint middle = min(pair_start + run_size, (uint)n);
    const uint pair_end = min(pair_start + 2 * run_size, (uint)n);
    const bool in_left = i < middle;
    const uint value = input_data[i];

    // Для левого элемента считаем строго меньшие элементы справа (lower_bound),
    // для правого — меньшие или равные слева (upper_bound).
    // При равенстве левый участок идёт первым: позиции уникальны, слияние устойчиво.
    uint low = in_left ? middle : pair_start;
    uint high = in_left ? pair_end : middle;
    while (low < high) {
        const uint mid = low + (high - low) / 2;
        const uint other = input_data[mid];
        if (other < value || (!in_left && other == value))
            low = mid + 1;
        else
            high = mid;
    }

    // Позиция = начало пары + свой номер в участке + число предшественников из другого.
    output_data[i + low - middle] = value;
}
