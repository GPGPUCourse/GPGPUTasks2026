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
    if (i >= (uint)n)
        return;

    const uint width = (uint)sorted_k;
    const uint start = i / (2 * width) * (2 * width);
    const uint middle = min(start + width, (uint)n);
    const uint end = min(start + 2 * width, (uint)n);
    const uint value = input_data[i];
    const bool left = i < middle;
    const uint other_start = left ? middle : start;
    uint low = other_start;
    uint high = left ? end : middle;

    while (low < high) {
        const uint mid = low + (high - low) / 2;
        if (input_data[mid] < value || (!left && input_data[mid] == value))
            low = mid + 1;
        else
            high = mid;
    }

    const uint offset = i - (left ? start : middle);
    output_data[start + offset + low - other_start] = value;
}
