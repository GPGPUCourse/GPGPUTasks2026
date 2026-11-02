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
    const uint i = get_global_id(0);
    if (i >= (uint)n)
        return;

    const uint width = (uint)sorted_k;
    const uint begin = (i / (2u * width)) * (2u * width);
    const uint middle = min(begin + width, (uint)n);
    const uint end = min(begin + 2u * width, (uint)n);
    const bool left = i < middle;
    const uint other_begin = left ? middle : begin;
    uint low = other_begin;
    uint high = left ? end : middle;
    const uint value = input_data[i];

    while (low < high) {
        const uint mid = low + (high - low) / 2u;
        const uint other = input_data[mid];
        if (other < value || (!left && other == value))
            low = mid + 1u;
        else
            high = mid;
    }

    const uint own_rank = i - (left ? begin : middle);
    output_data[begin + own_rank + (low - other_begin)] = value;
}
