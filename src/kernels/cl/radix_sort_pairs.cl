#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// LSD radix sort пар (key, value) - тот же алгоритм, что в task05, только при scatter вместе с ключом
// переезжает и значение. Сортировка стабильна (ранг внутри группы считается слева направо),
// поэтому треугольники с одинаковым Morton-кодом остаются в порядке возрастания исходного индекса.
//
// Один проход по RADIX_BITS битам начиная с bit_shift:
//   1) radix_sort_local_counting      - гистограмма цифр каждой группы, bucket-major: counts[bucket * num_groups + group]
//   2) radix_sort_scan_sum_reduction  - строит очередной уровень пирамиды сумм для scan массива counts
//   3) radix_sort_scan_accumulation   - добавляет в inclusive scan вклад одного уровня пирамиды
//   4) radix_sort_scatter_pairs       - каждый элемент пишет (key, value) на позицию offset(bucket, group) + rank

// Обнуляет буфер на GPU. Нужен перед каждым проходом: scan_accumulation накапливает результат через +=.
// (buffer.fill(0) с хоста шёл бы по PCI-E - намного медленнее.)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_fill_zeros(__global uint* buffer, unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
        buffer[i] = 0;
}

// Шаг 1 прохода. Каждая группа считает в локальной памяти, сколько её ключей попало в каждую из
// NUM_BUCKETS корзин по текущей цифре (key >> bit_shift) & RADIX_MASK, и выгружает гистограмму
// в counts в bucket-major порядке: counts[bucket * num_groups + group].
// Такой порядок нужен, чтобы exclusive scan по counts сразу давал позицию, с которой группа пишет корзину.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_local_counting(
    __global const uint* keys,
    __global       uint* counts,
    unsigned int n,
    unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int group = get_group_id(0);
    const unsigned int num_groups = get_num_groups(0);

    __local uint local_counts[NUM_BUCKETS];
    if (lid < NUM_BUCKETS)
        local_counts[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n)
        atomic_inc(&local_counts[(keys[gid] >> bit_shift) & RADIX_MASK]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < NUM_BUCKETS)
        counts[lid * num_groups + group] = local_counts[lid];
}

// Шаг 2 прохода. Строит следующий уровень пирамиды сумм: next[i] = cur[2i] + cur[2i+1].
// Элемент уровня pow2 - сумма блока counts длины 2^pow2. Хост вызывает в цикле, пока уровень не станет длины 1.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= (n + 1) / 2)
        return;

    const unsigned int left = pow2_sum[2 * i];
    const unsigned int right = (2 * i + 1 < n) ? pow2_sum[2 * i + 1] : 0;
    next_pow2_sum[i] = left + right;
}

// Шаг 3 прохода. Добавляет в inclusive scan вклад уровня pow2: отрезок [0, i] раскладывается на блоки
// длины степеней двойки по битам числа i+1; если бит pow2 установлен - добавляем блок этого уровня,
// заканчивающийся ровно на i (его номер ((i+1) >> pow2) - 1). После всех уровней prefix[i] = sum counts[0..i].
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    if ((i + 1) & (1u << pow2))
        prefix_sum_accum[i] += pow2_sum[((i + 1) >> pow2) - 1];
}

// Шаг 4 прохода. Каждая пара (key, value) переезжает на итоговую позицию:
//   offset(корзина, группа) = exclusive scan = prefix[idx] - counts[idx]
//   + rank = сколько элементов этой же группы с той же цифрой стоит левее.
// Rank сохраняет исходный порядок равных цифр (стабильность) - без неё LSD radix sort не работает,
// а для LBVH это ещё и даёт детерминированный порядок треугольников с одинаковым Morton-кодом.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scatter_pairs(
    __global const uint* keys,
    __global const uint* values,
    __global const uint* counts,
    __global const uint* prefix_sum, // inclusive scan массива counts
    __global       uint* sorted_keys,
    __global       uint* sorted_values,
    unsigned int n,
    unsigned int bit_shift)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int group = get_group_id(0);
    const unsigned int num_groups = get_num_groups(0);

    __local uint local_digits[GROUP_SIZE];

    const uint key = (gid < n) ? keys[gid] : 0;
    // Нити за пределами n получают несуществующую корзину, чтобы не влиять на ранги
    const uint digit = (gid < n) ? ((key >> bit_shift) & RADIX_MASK) : NUM_BUCKETS;
    local_digits[lid] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= n)
        return;

    // Ранг среди элементов группы с той же цифрой, стоящих левее - сохраняет стабильность
    uint rank = 0;
    for (unsigned int j = 0; j < lid; ++j)
        rank += (local_digits[j] == digit);

    const unsigned int idx = digit * num_groups + group;
    const uint offset = prefix_sum[idx] - counts[idx];
    sorted_keys[offset + rank] = key;
    sorted_values[offset + rank] = values[gid];
}
