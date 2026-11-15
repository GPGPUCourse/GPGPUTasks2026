#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

size_t lower_bound(__global const uint* input_data, size_t l, size_t r, unsigned int x);
size_t upper_bound(__global const uint* input_data, size_t l, size_t r, unsigned int x);

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    
    if (i >= n) {
        return;
    }
    unsigned int pair_size = 2 * sorted_k;
    unsigned int base = (i / pair_size) * pair_size;
    unsigned int offset = i - base;
    unsigned int local_idx = offset;
    bool is_left = 1;
    unsigned int other_idx = base + sorted_k;
    if (offset >= sorted_k) {
        local_idx = offset - sorted_k;
        is_left = 0;
        other_idx = base;
    }
    unsigned int other_len;
    if (other_idx >= n) { other_len = 0; }
    else { other_len = (sorted_k < n - other_idx ? sorted_k : n - other_idx); }

    size_t rank = 
    (is_left ? lower_bound(input_data, other_idx, other_idx + other_len, input_data[i]) - other_idx : 
    upper_bound(input_data, other_idx, other_idx + other_len, input_data[i]) - other_idx);

    size_t dst = base + rank + local_idx;
    output_data[dst] = input_data[i];
}

size_t lower_bound(__global const uint* input_data, size_t l, size_t r, unsigned int x) {
    while (l < r) {
        size_t mid = (l + r)/2;
        if (x > input_data[mid]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

size_t upper_bound(__global const uint* input_data, size_t l, size_t r, unsigned int x) {
    while (l < r) {
        size_t mid = (l + r)/2;
        if (x >= input_data[mid]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return r;
}