#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define ITEMS_PER_THREAD 4u

// Функция lower_bound для работы с глобальной памятью
uint lower_bound(__global const uint* arr, uint size, uint value) {
    uint left = 0;
    uint right = size; // Интервал [left, right)

    while (left < right) {
        // Защита от переполнения left + right для больших массивов
        uint mid = left + ((right - left) >> 1);

        if (arr[mid] < value) {
            left = mid + 1; // Искомый элемент находится правее mid
        } else {
            right = mid;    // Искомый элемент равен mid или левее mid
        }
    }
    return left; // Возвращает индекс первого элемента >= value (или size, если такого нет)
}

    // Возвращает индекс первого элемента, который строго больше, чем value.
// Если таких элементов нет, возвращает размер массива (size).
uint upper_bound(__global const uint* arr, uint size, uint value) {
    uint low = 0;
    uint high = size;

    while (low < high) {
        uint mid = low + (high - low) / 2;

        if (value >= arr[mid]) {
            low = mid + 1; // Искомый элемент находится правее mid
        } else {
            high = mid;    // Искомый элемент это mid или левее
        }
    }
    return low;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
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

    unsigned int k = sorted_k;

    const uint block_size = 2u * k;
    const uint base = i & ~(block_size - 1u);

    const uint count = (uint)n;
    uint mid = min(base + k, count);
    uint end = min(base + (2u * k), count);
    unsigned int pos = 0;
    const uint x = input_data[i];

    if (i < mid) {
        pos = i + lower_bound(input_data + mid, end - mid, x);
    } else {
        pos = base + (i - mid) + upper_bound(input_data + base, mid - base, x);
    }

    output_data[pos] = x;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void local_sort(
    __global const  uint* input_data,
    __global        uint* output_data,
                    int n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint tile[GROUP_SIZE];

    tile[lid] = gid < (uint)n ? input_data[gid] : 0xffffffffu;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint span = 2; span <= GROUP_SIZE; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {

            if ((lid & stride) == 0) {
                const uint partner = lid ^ stride;

                const uint a = tile[lid];
                const uint b = tile[partner];

                const bool ascending = ((lid & span) == 0);
                if (ascending) {
                    tile[lid] = min(a, b);
                    tile[partner] = max(a, b);
                } else {
                    tile[lid] = max(a, b);
                    tile[partner] = min(a, b);
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (gid < (uint)n) {
        output_data[gid] = tile[lid];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_tiled(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const uint gid = get_global_id(0);
    const uint N = (uint)n;

    // Каждый поток записывает до 4 элементов результата
    const uint out_begin = gid * ITEMS_PER_THREAD;
    if (out_begin >= N) return;

    const uint k = (uint)sorted_k;
    const uint pair_size = 2u * k;

    // Находим пару отсортированных подмассивов
    const uint base = out_begin & ~(pair_size - 1u);
    const uint mid = min(base + k, N);
    const uint end = min(base + pair_size, N);

    const uint lenA = mid - base;
    const uint lenB = end - mid;

    // Позиция начала нашего участка внутри результата слияния
    const uint diag = out_begin - base;

    // Находим разбиение: a + b = diag
    uint low = diag > lenB ? diag - lenB : 0u;
    uint high = min(diag, lenA);

    while (low < high) {
        const uint a = low + ((high - low) >> 1);
        const uint b = diag - a;

        // При равенстве элементы A должны идти раньше B
        if (b > 0u && a < lenA &&
            input_data[mid + b - 1u] >= input_data[base + a]) {
            low = a + 1u;
        } else {
            high = a;
        }
    }

    uint a = low;
    uint b = diag - a;

    // Последовательно формируем до 4 элементов результата
    const uint out_end = min(out_begin + ITEMS_PER_THREAD, end);

    for (uint out = out_begin; out < out_end; ++out) {
        const uint va = a < lenA
            ? input_data[base + a] : 0xffffffffu;

        const uint vb = b < lenB
            ? input_data[mid + b] : 0xffffffffu;

        const bool takeA =
            a < lenA && (b >= lenB || va <= vb);

        output_data[out] = takeA ? va : vb;

        a += (uint)takeA;
        b += (uint)(!takeA);
    }
}