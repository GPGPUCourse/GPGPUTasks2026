#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Иерархическое сведение гистограмм: групп слишком много (n / GROUP_SIZE ~ сотни тысяч),
// поэтому один work-item, суммирующий колонку целиком, становится узким местом (сотни тысяч итераций).
// Группы разбиваем на супер-группы по 256 штук. Сведение одно и то же на обоих уровнях -
// "256 строк таблицы -> 1 строка" (гистограмма супер-группы), поэтому кернел один, два запуска:
//  - этап 1: из гистограмм групп (group_histograms) получается гистограмма супер-группы:
//    item (super_group, digit) суммирует колонку digit по 256 группам своей супер-группы.
//  - этап 2: из гистограмм супер-групп получается гистограмма всего массива (одна строка):
//    item digit суммирует колонку digit по всем super_group_count супер-группам.
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* row_histograms,

    __global       uint* row_sums,

    unsigned int row_count,
    unsigned int mode)
{
    const unsigned int idx = get_global_id(0);

    if (mode == 0) {
        // item'ов ровно 256 * row_count: idx = super_group * 256 + digit
        const unsigned int super_group = idx / 256;
        const unsigned int digit = idx % 256;

        const unsigned int row_end = min((super_group + 1) * 256, row_count);

        uint digit_count_in_super_group = 0;
        for (unsigned int row = super_group * 256; row < row_end; ++row) {
            digit_count_in_super_group += row_histograms[row * 256 + digit];
        }

        row_sums[idx] = digit_count_in_super_group;
    } else {
        // item'ов ровно 256: idx = digit, суммируем колонку по всем супер-группам
        const unsigned int digit = idx;
        uint digit_count_in_array = 0;
        for (unsigned int row = 0; row < row_count; ++row) {
            digit_count_in_array += row_histograms[row * 256 + digit];
        }
        row_sums[digit] = digit_count_in_array;
    }
}
