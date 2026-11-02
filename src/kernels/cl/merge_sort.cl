#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint binary_search(__global const uint* input_data, uint start, uint end, uint i)
{
    const uint value = input_data[i];
    while (start < end) {
        const uint mid = start + (end - start) / 2;
        const uint other = input_data[mid];
        if (other < value || (other == value && mid < i))
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    const uint pair_start = (i / (2 * sorted_k)) * (2 * sorted_k);
    const uint middle = min(pair_start + sorted_k, (uint)n);
    const uint pair_end = min(middle + sorted_k, (uint)n);
    const bool is_left = i < middle;
    const uint own_start = is_left ? pair_start : middle;
    const uint other_start = is_left ? middle : pair_start;
    const uint other_end = is_left ? pair_end : middle;
    const uint pos = binary_search(input_data, other_start, other_end, i);
    const uint dest = pair_start + (i - own_start) + (pos - other_start);
    output_data[dest] = input_data[i];
}
