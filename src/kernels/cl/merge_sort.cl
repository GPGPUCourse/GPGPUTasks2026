#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

int merge_rank_local(int k, __local const uint* a, int na,
                     __local const uint* b, int nb)
{
    int lo = max(0, k - nb), hi = min(k, na);
    while (lo < hi) {
        // Since i < min(k, na), both a[i] and b[k - i - 1] exist.
        int i = lo + (hi - lo) / 2;
        if (b[k - i - 1] >= a[i])
            lo = i + 1;
        else
            hi = i;
    }
    return lo;
}

int merge_rank_global(int k, __global const uint* a, int na,
                      __global const uint* b, int nb)
{
    int lo = max(0, k - nb), hi = min(k, na);
    while (lo < hi) {
        int i = lo + (hi - lo) / 2;
        if (b[k - i - 1] >= a[i])
            lo = i + 1;
        else
            hi = i;
    }
    return lo;
}

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort_local(__global const uint* input_data,
                             __global uint* output_data, int n)
{
    const int lid = get_local_id(0);
    const int base = get_group_id(0) * MERGE_BLOCK_SIZE;
    __local uint first[MERGE_BLOCK_SIZE];
    __local uint second[MERGE_BLOCK_SIZE];
    for (int t = lid; t < MERGE_BLOCK_SIZE; t += MERGE_GROUP_SIZE)
        first[t] = base + t < n ? input_data[base + t] : 0xffffffffu;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint values[MERGE_ITEMS_PER_THREAD];
    for (int t = 0; t < MERGE_ITEMS_PER_THREAD; ++t)
        values[t] = first[lid * MERGE_ITEMS_PER_THREAD + t];
#define SORT_PAIR(a, b) { uint lo = min(values[a], values[b]); \
                         values[b] = max(values[a], values[b]); values[a] = lo; }
    SORT_PAIR(0,1); SORT_PAIR(2,3); SORT_PAIR(4,5); SORT_PAIR(6,7);
    SORT_PAIR(0,2); SORT_PAIR(1,3); SORT_PAIR(4,6); SORT_PAIR(5,7);
    SORT_PAIR(1,2); SORT_PAIR(5,6); SORT_PAIR(0,4); SORT_PAIR(3,7);
    SORT_PAIR(1,5); SORT_PAIR(2,6);
    SORT_PAIR(1,4); SORT_PAIR(3,6);
    SORT_PAIR(2,4); SORT_PAIR(3,5);
    SORT_PAIR(3,4);
#undef SORT_PAIR
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = 0; t < MERGE_ITEMS_PER_THREAD; ++t)
        first[lid * MERGE_ITEMS_PER_THREAD + t] = values[t];
    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint* src = first;
    __local uint* dst = second;
    for (int width = MERGE_ITEMS_PER_THREAD; width < MERGE_BLOCK_SIZE; width *= 2) {
        int offset = lid * MERGE_ITEMS_PER_THREAD;
        int pair = offset / (2 * width) * (2 * width);
        int diagonal = offset - pair;
        __local const uint* a = src + pair;
        __local const uint* b = a + width;
        int i = merge_rank_local(diagonal, a, width, b, width);
        int j = diagonal - i;
        #pragma unroll
        for (int t = 0; t < MERGE_ITEMS_PER_THREAD; ++t) {
            bool take_a = j == width || (i < width && a[i] <= b[j]);
            dst[offset + t] = take_a ? a[i++] : b[j++];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }
    for (int t = lid; t < MERGE_BLOCK_SIZE; t += MERGE_GROUP_SIZE)
        if (base + t < n)
            output_data[base + t] = src[t];
}

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort_partitions(__global const uint* input_data,
                                   __global uint* partitions,
                                   int sorted_k, int n)
{
    const int t = get_global_id(0);
    const int begin = t * MERGE_BLOCK_SIZE;
    if (begin >= n)
        return;
    const int pair = begin / (2 * sorted_k) * (2 * sorted_k);
    const int middle = min(pair + sorted_k, n);
    const int end = min(pair + 2 * sorted_k, n);
    partitions[t] = merge_rank_global(begin - pair, input_data + pair,
                                     middle - pair, input_data + middle, end - middle);
}

__attribute__((reqd_work_group_size(MERGE_GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
    __global const uint* partitions,
                   int  sorted_k,
                   int  n)
{
    const int lid = get_local_id(0);
    const int group = get_group_id(0);
    const int output_begin = group * MERGE_BLOCK_SIZE;
    const int pair = output_begin / (2 * sorted_k) * (2 * sorted_k);
    const int middle = min(pair + sorted_k, n);
    const int end = min(pair + 2 * sorted_k, n);
    const int na = middle - pair, nb = end - middle;
    const int diagonal = output_begin - pair;
    const int count = min(MERGE_BLOCK_SIZE, n - output_begin);
    __local uint tile[MERGE_BLOCK_SIZE];
    __local uint result[MERGE_BLOCK_SIZE];

    const int a0 = partitions[group];
    const int b0 = diagonal - a0;
    // At the end of a pair the next partition belongs to another pair.
    const int a1 = diagonal + count == na + nb ? na : partitions[group + 1];
    const int a_count = a1 - a0;
    const int b_count = count - a_count;
    for (int t = lid; t < a_count; t += MERGE_GROUP_SIZE)
        tile[t] = input_data[pair + a0 + t];
    for (int t = lid; t < b_count; t += MERGE_GROUP_SIZE)
        tile[a_count + t] = input_data[middle + b0 + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    int k = min(lid * MERGE_ITEMS_PER_THREAD, count);
    int i = merge_rank_local(k, tile, a_count, tile + a_count, b_count);
    int j = k - i;
    #pragma unroll
    for (int t = 0; t < MERGE_ITEMS_PER_THREAD; ++t) {
        if (k + t >= count)
            break;
        bool take_a = j == b_count || (i < a_count && tile[i] <= tile[a_count + j]);
        // Spread consecutive eight-element runs over the local memory banks.
        result[(k + t) ^ (((k + t) >> 5) & 7)] = take_a ? tile[i++] : tile[a_count + j++];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = lid; t < count; t += MERGE_GROUP_SIZE)
        output_data[output_begin + t] = result[t ^ ((t >> 5) & 7)];
}
