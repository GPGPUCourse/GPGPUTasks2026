#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  len, // power of 2, len of each of current sorted parts
                   int  n)
{
    const unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    uint result_len = len * 2;
    uint a_begin = (i / result_len) * result_len;
    __global const uint* a = input_data + a_begin;
    if (a_begin + len >= n) {
        output_data[i] = input_data[i];
        return;
    }
    __global const uint* b = a + len;
    uint a_len = len;
    uint b_len = min((uint)len, (uint)n - (a_begin + (uint)len));

    bool swapped = false;
    if (a_begin + len <= i) {
        swapped = true;
        __global const uint* q = a;
        a = b;
        b = q;

        uint w = a_len;
        a_len = b_len;
        b_len = w;
    }

    uint value_to_find = input_data[i];
    uint l = 0;
    uint r = b_len;
    while (l < r) {
        uint m = l + (r - l) / 2;
        if (b[m] < value_to_find || (swapped && b[m] <= value_to_find)) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    uint our_index = i - (uint)(a - input_data);
    uint result_i = a_begin + l + our_index;
    output_data[result_i] = value_to_find;
}
