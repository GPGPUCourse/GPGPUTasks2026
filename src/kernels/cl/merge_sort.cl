#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define TILE GROUP_SIZE

uint lower_bound(__local const uint* data, uint l, uint r, int x){
    while (l < r){
        uint m = l + (r - l) / 2;
        if (data[m] < x) l = m + 1;
        else r = m;
    }
    return l;
}

uint upper_bound(__local const uint* data, uint l, uint r, int x){
    while (l < r){
        uint m = l + (r - l) / 2;
        if (data[m] <= x) l = m + 1;
        else r = m;
    }
    return l;
}

uint find_border(__global const uint* data, uint left_start, uint right_start, uint len_a, uint len_b, uint d){
    uint l = d > len_b ? d - len_b : 0;
    uint r = min(d, len_a);
    while (l < r){
        uint a = l + (r - l) / 2;
        uint b = d - a;
        if (b > 0 && a < len_a && data[right_start + b-1] >= data[left_start + a]) l = a + 1;
        else r = a;
    }
    return l;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint local_id = get_local_id(0);
    const uint group = get_group_id(0);
    const uint pair_span = 2u * (uint)sorted_k;
    if (2u * (uint)sorted_k < TILE){
        const uint global_id = get_global_id(0);
        const uint group_start = TILE * group;
        const uint valid = min((uint)TILE, (uint)n - group_start);
        if (valid <= 0) return;
        const uint pair_start = local_id / pair_span * pair_span;

        __local uint data[TILE];
        if (global_id < (uint)n){
            data[local_id] = input_data[global_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (global_id < (uint)n) {
            const uint value = data[local_id];
            const uint left_end  = min(pair_start + (uint)sorted_k, valid);
            const uint right_end = min(pair_start + pair_span, valid);

            uint offset_in_pair;

            if (local_id < left_end) {
                uint pos_in_right = lower_bound(data, left_end, right_end, value);
                offset_in_pair = (local_id - pair_start) + (pos_in_right - left_end);
            } else {
                uint pos_in_left = upper_bound(data, pair_start, left_end, value);
                offset_in_pair = (local_id - left_end) + (pos_in_left - pair_start);
            }

            output_data[group_start + pair_start + offset_in_pair] = value;
        }
        return;
    }
    const uint group_per_pair = (pair_span + TILE - 1u) / TILE;
    const uint pair_id = group / group_per_pair;
    const uint pair_start = pair_id * pair_span;
    if (pair_start >= (uint)n) return;

    const uint tile_id = group % group_per_pair;
    const uint len_a = min((uint)sorted_k, (uint)n - pair_start);
    const uint right_start = pair_start + len_a;
    const uint len_b = min((uint)sorted_k, (uint)n - right_start);
    const uint pair_length = len_a + len_b;
    const uint d_start = tile_id * TILE;
    if (d_start >= pair_length) return;
    const uint d_end = min(pair_length, d_start + TILE);


    __local uint borders[2];
    if (local_id == 0){
        borders[0] = find_border(input_data, pair_start, right_start, len_a, len_b, d_start);
        borders[1] = find_border(input_data, pair_start, right_start, len_a, len_b, d_end);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint a_start = borders[0];
    uint a_end = borders[1];
    uint b_start = d_start - a_start;
    uint b_end = d_end - a_end;

    __local uint sorted_data[TILE];
    const uint count_a = a_end - a_start;
    const uint count_b = b_end - b_start;

    if (local_id < count_a) {
        sorted_data[local_id] = input_data[pair_start + a_start + local_id];
    } else if (local_id < count_a + count_b) {
        const uint b_index = local_id - count_a;
        sorted_data[local_id] = input_data[right_start + b_start + b_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < count_a){
        uint value = sorted_data[local_id];
        uint pos_in_b = lower_bound(sorted_data, count_a, count_a + count_b, value);
        uint offset = local_id + (pos_in_b - count_a);
        output_data[pair_start + d_start + offset] = value;
    } else if (local_id < count_a + count_b){
        uint value = sorted_data[local_id];
        uint pos_in_a = upper_bound(sorted_data, 0, count_a, value);
        uint offset = local_id - count_a + pos_in_a;
        output_data[pair_start + d_start + offset] = value;
    }
}
