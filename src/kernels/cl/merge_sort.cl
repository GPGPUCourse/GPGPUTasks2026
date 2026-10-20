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
    // TODO
    if (i >= n) {
        return;
    }
    const uint start = i / sorted_k * sorted_k;
    const uint other = start ^ (uint)sorted_k;
    uint left = other;
    uint right = min(other + sorted_k, (uint)n);
    const uint value = input_data[i];
    while (left < right) {
        const uint mid = left + (right - left) / 2;
        if (input_data[mid] < value || (other < start && input_data[mid] == value)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    output_data[min(start, other) + i - start + left - other] = value;
}
