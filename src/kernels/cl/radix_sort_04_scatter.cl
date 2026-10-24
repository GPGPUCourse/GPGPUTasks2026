#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sums,
    __global       uint* output,
    unsigned int iter_num,
    unsigned int elems_per_group,
    unsigned int n)
{
    __local uint digits[GROUP_SIZE];
    __local uint group_base[RADIX_NUM_MASK];

    const unsigned int local_ind = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int num_groups = get_num_groups(0);
    const unsigned int shift = iter_num * RADIX_BITS;

    const unsigned int group_start = elems_per_group * group_id;
    const unsigned int group_end = group_start + elems_per_group < n ? group_start + elems_per_group : n;

    if (local_ind < RADIX_NUM_MASK) {
        const unsigned int idx = local_ind * num_groups + group_id;
        group_base[local_ind] = idx == 0 ? 0 : prefix_sums[idx - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int tile = group_start; tile < group_end; tile += GROUP_SIZE) {
        const unsigned int i = tile + local_ind;
        const bool valid = i < group_end;

        unsigned int x = 0;
        unsigned int digit = RADIX_NUM_MASK;
        if (valid) {
            x = input[i];
            digit = (x >> shift) & (RADIX_NUM_MASK - 1);
        }
        digits[local_ind] = digit;
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int rank = 0;
        unsigned int total = 0;
        if (valid) {
            for (unsigned int j = 0; j < GROUP_SIZE; ++j) {
                const unsigned int same = digits[j] == digit ? 1 : 0;
                total += same;
                if (j < local_ind) {
                    rank += same;
                }
            }
            output[group_base[digit] + rank] = x;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (valid && rank + 1 == total) {
            group_base[digit] += total;
        }
    }
}