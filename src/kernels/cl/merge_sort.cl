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
    const unsigned int i = get_global_id(0);
    if (i >= (unsigned int) n)
        return;

    const unsigned int k = sorted_k;
    const unsigned int base = (i / (2 * k)) * (2 * k);
    const unsigned int mid = min(base + k, (unsigned int) n);
    const unsigned int end = min(base + 2 * k, (unsigned int) n);

    const uint x = input_data[i];
    const bool is_left = i < mid;

    unsigned int lo = is_left ? mid : base;
    unsigned int hi = is_left ? end : mid;
    const unsigned int other_begin = lo;
    while (lo < hi) {
        const unsigned int m = lo + ((hi - lo) >> 1);
        const uint y = input_data[m];
        const bool go_right = is_left ? (y < x) : (y <= x);
        if (go_right)
            lo = m + 1;
        else
            hi = m;
    }
    const unsigned int cnt = lo - other_begin;
    const unsigned int own_idx = is_left ? (i - base) : (i - mid);

    const unsigned int out_idx = base + own_idx + cnt;
    output_data[out_idx] = x;
}
