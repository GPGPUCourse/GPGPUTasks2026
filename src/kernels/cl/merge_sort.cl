#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint lower_bound(__global const uint* data, uint l, uint r, int x){
    while (l < r){
        uint m = l + (r - l) / 2;
        if (data[m] < x) l = m + 1;
        else r = m;
    }
    return l;
}

uint upper_bound(__global const uint* data, uint l, uint r, int x){
    while (l < r){
        uint m = l + (r - l) / 2;
        if (data[m] <= x) l = m + 1;
        else r = m;
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
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    const uint value = input_data[i];

    const uint pair_start = (i / (2 * sorted_k)) * (2 * sorted_k);
    const uint left_end = min(pair_start + sorted_k, (uint)n);
    const uint right_end = min(pair_start + 2 * sorted_k, (uint)n);
    const bool is_in_left = i < left_end;

    if (is_in_left){
        uint pos_in_right = lower_bound(input_data, left_end, right_end, value);
        output_data[i + (pos_in_right - left_end)] = value;
    } else{
        uint pos_in_left = upper_bound(input_data, pair_start, left_end, value);
        output_data[pos_in_left + i - left_end] = value;
    }
}
