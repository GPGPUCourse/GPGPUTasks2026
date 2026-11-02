#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

void merge_sort_local(__local const uint* input_data, __local uint* output_data, uint i, int sorted_k, int n) {
    if (i >= n) {
        return;
    }

    const uint size = sorted_k * 2;
    const uint start = (i / size) * size;

    const uint l_start = start;
    const uint l_end = min(start + (uint)sorted_k, (uint)n);
    const uint r_start = l_end;
    const uint r_end = min(start + size, (uint)n);
    const uint is_left = l_start <= i && i < l_end;

    const uint elem = input_data[i];
    int l = (int)(is_left ? r_start : l_start) - 1;
    int r = is_left ? r_end : l_end;
    while (r - l > 1) {
        const int m = l + (r - l) / 2;
        if (input_data[m] < elem || (!is_left && input_data[m] == elem)) {
            l = m;
        } else {
            r = m;
        }
    }

    const uint less_cnt = is_left ? r - r_start : r - l_start;
    const uint my_start = is_left ? l_start : r_start;
    const uint target_idx = start + (i - my_start) + less_cnt;
    output_data[target_idx] = elem;
}

void merge_sort_global(__global const uint* input_data, __global uint* output_data, uint i, int sorted_k, int n) {
    if (i >= n) {
        return;
    }

    const uint size = sorted_k * 2;
    const uint start = (i / size) * size;

    const uint l_start = start;
    const uint l_end = min(start + (uint)sorted_k, (uint)n);
    const uint r_start = l_end;
    const uint r_end = min(start + size, (uint)n);
    const uint is_left = l_start <= i && i < l_end;

    const uint elem = input_data[i];
    int l = (int)(is_left ? r_start : l_start) - 1;
    int r = is_left ? r_end : l_end;
    while (r - l > 1) {
        const int m = l + (r - l) / 2;
        if (input_data[m] < elem || (!is_left && input_data[m] == elem)) {
            l = m;
        } else {
            r = m;
        }
    }

    const uint less_cnt = is_left ? r - r_start : r - l_start;
    const uint my_start = is_left ? l_start : r_start;
    const uint target_idx = start + (i - my_start) + less_cnt;
    output_data[target_idx] = elem;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    __local uint buf1[GROUP_SIZE];
    __local uint buf2[GROUP_SIZE];

    // fused
    if (sorted_k < GROUP_SIZE) {
        const uint block_idx = get_local_id(0);
        const uint i = get_global_id(0);
        buf1[block_idx] = i < n ? input_data[i] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        uint is_first = 1;
        while (sorted_k * 2 <= GROUP_SIZE) {
            if (is_first) {
                merge_sort_local(buf1, buf2, block_idx, sorted_k, GROUP_SIZE);
            } else {
                merge_sort_local(buf2, buf1, block_idx, sorted_k, GROUP_SIZE);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            is_first = (is_first + 1) & 1;
            sorted_k *= 2;
        }

        if (i < n) {
            output_data[i] = is_first ? buf1[block_idx] : buf2[block_idx];
        }

        return;
    }

    merge_sort_global(input_data, output_data, get_global_id(0), sorted_k, n);
}
