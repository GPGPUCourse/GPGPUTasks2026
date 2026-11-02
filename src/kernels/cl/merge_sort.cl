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
    if (i < n) {
        size_t block_size = 2 * (size_t)sorted_k;
        size_t block_start = i / block_size * block_size;
        size_t middle = min(block_start + (size_t)sorted_k, (size_t)n);
        size_t block_end = min(middle + (size_t)sorted_k, (size_t)n);
        bool from_left = i < middle;
        uint value = input_data[i];

        size_t left = from_left ? middle : block_start;
        size_t right = from_left ? block_end : middle;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            uint candidate = input_data[mid];
            if (candidate < value || (!from_left && candidate == value)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        size_t local_index = i - (from_left ? block_start : middle);
        size_t other_prefix = left - (from_left ? middle : block_start);
        output_data[block_start + local_index + other_prefix] = value;
    }
}
