#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k, // starts from 1
                   uint  n)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) return;

    const uint block_index = gid / (sorted_k * 2);
    const uint local_i = gid % (sorted_k * 2);
    const uint block_start = block_index * (sorted_k * 2);
    __global const uint* merge_input = input_data + block_start;
    __global uint* merge_output = output_data + block_start;

    const uint sorted_left = sorted_k;
    if (block_start + sorted_left >= n) {
        output_data[gid] = input_data[gid]; // tail is already sorted
        return;
    }
    const uint sorted_right = min(sorted_k, n - block_start - sorted_left);
    
    // merge -----------------------------------------
    
    bool in_left = local_i < sorted_left;

    const uint our_value = merge_input[local_i];
    __global const uint* ptr_binsearch = in_left ? merge_input + sorted_left : merge_input;

    uint l = 0;
    uint r = in_left ? sorted_right : sorted_left;
    while (l < r) {
        uint m = (l + r) / 2;
        if (in_left ? ptr_binsearch[m] < our_value : ptr_binsearch[m] <= our_value) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    const uint our_offset = in_left ? local_i : local_i - sorted_left;
    const uint binsearch_offset = l;

    merge_output[our_offset + binsearch_offset] = our_value;
    
    // merge -----------------------------------------
}
