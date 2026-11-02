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
