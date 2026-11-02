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
    if (i >= (size_t)n) {
        return;
    }

    const size_t start = i / (2 * (size_t)sorted_k) * (2 * (size_t)sorted_k);
    const size_t middle = min(start + sorted_k, (size_t)n);

    const size_t rank = i - start;
    const size_t l_size = middle - start;
    const size_t r_size = min(middle + sorted_k, (size_t)n) - middle;

    size_t low = rank > r_size ? rank - r_size : 0;
    size_t high = min(rank, l_size);
    while (low < high) {
        const size_t l = (low + high) / 2;
        const size_t r = rank - l;
        if (r > 0 && input_data[middle + r - 1] >= input_data[start + l]) {
            low = l + 1;
        } else {
            high = l;
        }
    }

    const size_t l = low;
    const size_t r = rank - l;
    if (l < l_size && (r == r_size || input_data[start + l] <= input_data[middle + r])) {
        output_data[i] = input_data[start + l];
    } else {
        output_data[i] = input_data[middle + r];
    }
}
