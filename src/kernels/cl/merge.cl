#include "../defines.h"
#include "helpers/rassert.cl"

// Returns how many elements come from the left
// among the first d elements of the merged result
inline uint merge_cut(
    __global const uint* x,
    uint a, uint na,
    uint b, uint nb,
    uint d)
{
    const uint min_i = d > nb ? d - nb : 0u;
    const uint max_i = min(d, na);

    int l = -1;
    int r = max_i;

    while (l < r - 1) {
        const int m = (l + r) / 2;
        const uint i = m;
        const uint j = d - i;

        if (i < min_i || (j > 0u && i < na && x[b + j - 1u] >= x[a + i]))
            l = m;
        else
            r = m;
    }

    return r;
}

// Returns how many elements come from the left
// among the first d elements of the merged result
inline uint local_cut(
    __local const uint* a, uint na,
    __local const uint* b, uint nb,
    uint d)
{
    uint min_i = d > nb ? d - nb : 0u;
    uint max_i = min(d, na);

    int l = -1;
    int r = max_i;

    while (l < r - 1) {
        int m = (l + r) / 2;
        int i = m;
        int j = d - i;

        if (i < min_i || (j > 0u && i < na && b[j - 1u] >= a[i])) {
            l = m;
        } else {
            r = m;
        }
    }

    return r;
}

#define ITEMS_PER_THREAD 4u
#define ITEMS_PER_GROUP (GROUP_SIZE * ITEMS_PER_THREAD)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge(
    __global const uint* input_data,
    __global uint* output_data,
    uint sorted_k,
    uint n)
{
    const uint li = get_local_id(0);

    const uint span = 2u * sorted_k;
    const uint groups_per_merge = (span + ITEMS_PER_GROUP - 1u) / ITEMS_PER_GROUP;

    const uint group = get_group_id(0);
    const uint pair = group / groups_per_merge;
    const uint tile = group % groups_per_merge;
    const uint a_from = pair * span;

    if (a_from >= n)
        return;

    const uint a_len = min(sorted_k, n - a_from);
    const uint b_from = a_from + a_len;
    const uint b_len = min(sorted_k, n - b_from);
    const uint len = a_len + b_len;

    const uint write_from = tile * ITEMS_PER_GROUP;

    if (write_from >= len)
        return;

    const uint write_to = min(write_from + ITEMS_PER_GROUP, len);

    const uint a_skip = merge_cut(
        input_data, a_from, a_len, b_from, b_len, write_from);
    const uint a_spare = merge_cut(
        input_data, a_from, a_len, b_from, b_len, write_to);

    const uint b_skip = write_from - a_skip;
    const uint b_spare = write_to - a_spare;

    const uint left_len = a_spare - a_skip;
    const uint right_len = b_spare - b_skip;

    __local uint pool[ITEMS_PER_GROUP];
    __local uint* left = pool;
    __local uint* right = pool + left_len;

#pragma unroll
    for (uint k = li; k < left_len; k += GROUP_SIZE)
        left[k] = input_data[a_from + a_skip + k];

#pragma unroll
    for (uint k = li; k < right_len; k += GROUP_SIZE)
        right[k] = input_data[b_from + b_skip + k];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint count = write_to - write_from;
    const uint out_from = li * ITEMS_PER_THREAD;

    if (out_from < count) {
        uint i = local_cut(left, left_len, right, right_len, out_from);
        uint j = out_from - i;
        uint values[ITEMS_PER_THREAD];

#pragma unroll
        for (uint k = 0;
            k < ITEMS_PER_THREAD && out_from + k < count;
            ++k) {
            if (i < left_len && (j >= right_len || left[i] <= right[j]))
                values[k] = left[i++];
            else
                values[k] = right[j++];
        }

        if (out_from + ITEMS_PER_THREAD <= count) {
            vstore4(
                (uint4)(values[0], values[1], values[2], values[3]),
                0,
                output_data + a_from + write_from + out_from);
        } else {
#pragma unroll
            for (uint k = 0; out_from + k < count; ++k)
                output_data[a_from + write_from + out_from + k] = values[k];
        }
    }
}