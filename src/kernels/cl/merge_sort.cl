#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

unsigned int lower_bound(__global const uint* input_data, const int input_sz, const unsigned int elem) {
    int l = -1, r = input_sz;
    while (l + 1 < r) {
        const int m = (l + r) / 2;
        if (input_data[m] < elem) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

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

    const unsigned int block_start = i - i % (2 * sorted_k);
    const unsigned int elem = input_data[i];
    int out_ind = i;

    if (i % (2 * sorted_k) < sorted_k) {
        const int neighbor_start = block_start + sorted_k;
        const int neighbor_sz = neighbor_start + sorted_k <= n ? sorted_k : n - neighbor_start;
        out_ind += neighbor_start < n ? lower_bound(input_data + neighbor_start, neighbor_sz, elem) : 0;

    } else {
        const int neighbor_start = block_start;
        const int neighbor_sz = sorted_k;
        out_ind += lower_bound(input_data + neighbor_start, neighbor_sz, elem + 1) - neighbor_sz;
    }

    output_data[out_ind] = elem;
}
