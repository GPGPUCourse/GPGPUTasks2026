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
    if (i >= n) {
        return;
    }

    uint width = sorted_k;
    uint base = i / (2 * width) * (2 * width);
    uint split = min(base + width, (uint)n);
    uint left = i < split ? split : base;
    uint right = i < split ? min(base + 2 * width, (uint)n) : split;
    uint value = input_data[i];

    while (left < right) {
        uint mid = left + (right - left) / 2;
        if (input_data[mid] < value || (i >= split && input_data[mid] == value)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    output_data[i + left - split] = value;
}
