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
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    const int block_size = 2 * sorted_k;
    const int block_start = i / block_size * block_size;
    const int middle = min(block_start + sorted_k, n);
    const int block_end = min(block_start + block_size, n);

    const bool from_left = i < middle;
    const uint value = input_data[i];

    int left = from_left ? middle : block_start;
    int right = from_left ? block_end : middle;

    while (left < right) {
        const int mid = left + (right - left) / 2;
        const uint candidate = input_data[mid];

        if (candidate < value || (!from_left && candidate == value)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // left - middle === колво эл-ов во 2 половине что должны быть левее нашего value
    output_data[i + left - middle] = value;
}
