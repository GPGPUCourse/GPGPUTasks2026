#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint i = get_global_id(0);
    if (i >= (uint)n) {
        return;
    }

    const uint run_size = (uint)sorted_k;
    const uint pair_size = run_size * 2u;
    const uint pair_begin = (i / pair_size) * pair_size;
    const uint middle = min(pair_begin + run_size, (uint)n);
    const uint pair_end = min(pair_begin + pair_size, (uint)n);
    const uint value = input_data[i];

    uint low;
    uint high;
    uint position;
    if (i < middle) {
        // Элементы левой половины ставим перед равными элементами правой:
        // ищем число элементов справа, которые строго меньше value.
        low = middle;
        high = pair_end;
        while (low < high) {
            const uint probe = low + (high - low) / 2u;
            if (input_data[probe] < value) {
                low = probe + 1u;
            } else {
                high = probe;
            }
        }
        position = pair_begin + (i - pair_begin) + (low - middle);
    } else {
        // Для правой половины считаем элементы слева, которые меньше либо
        // равны value. Такое правило сохраняет порядок при дубликатах.
        low = pair_begin;
        high = middle;
        while (low < high) {
            const uint probe = low + (high - low) / 2u;
            if (input_data[probe] <= value) {
                low = probe + 1u;
            } else {
                high = probe;
            }
        }
        position = pair_begin + (i - middle) + (low - pair_begin);
    }

    output_data[position] = value;
}
