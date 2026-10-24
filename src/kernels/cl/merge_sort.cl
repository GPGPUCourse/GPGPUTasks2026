#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Один проход bottom-up merge sort:
// массив уже состоит из отсортированных кусков длины sorted_k;
// сливаем соседние пары в куски длины 2*sorted_k.
//
// Нить gid отвечает за одну позицию выхода: бинарным поиском по "диагонали"
// находит, какой элемент из левого/правого куска должен туда попасть.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const int gid = get_global_id(0);
    if (gid >= n)
        return;

    const int pair = 2 * sorted_k;
    const int block_start = (gid / pair) * pair;
    const int mid = min(block_start + sorted_k, n);
    const int block_end = min(block_start + pair, n);

    // Правого куска нет — просто копируем.
    if (mid >= block_end) {
        output_data[gid] = input_data[gid];
        return;
    }

    const int left_len = mid - block_start;
    const int right_len = block_end - mid;
    const int diag = gid - block_start; // ранг внутри слитого блока

    // Сколько элементов левого куска входит в первые diag позиций слияния.
    int begin = max(0, diag - right_len);
    int end = min(diag, left_len);
    while (begin < end) {
        const int m = (begin + end) >> 1;
        const uint a = input_data[block_start + m];
        const uint b = input_data[mid + diag - 1 - m];
        // <= — стабильность: равные элементы берём сначала из левого куска.
        if (a <= b)
            begin = m + 1;
        else
            end = m;
    }

    const int a_idx = begin;
    const int b_idx = diag - begin;

    uint result;
    if (a_idx >= left_len)
        result = input_data[mid + b_idx];
    else if (b_idx >= right_len)
        result = input_data[block_start + a_idx];
    else {
        const uint a = input_data[block_start + a_idx];
        const uint b = input_data[mid + b_idx];
        result = (a <= b) ? a : b;
    }
    output_data[gid] = result;
}
