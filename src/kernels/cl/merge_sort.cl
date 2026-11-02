#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void small_blocks_sort(
    __global const uint* input_data,
    __global       uint* output_data)
{
    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);

    __local uint loc_mem[GROUP_SIZE];
    loc_mem[local_index] = input_data[global_index];

    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_block = 2;
    while (cur_block <= GROUP_SIZE) {
        int cur_len = cur_block / 2;

        while (cur_len >= 1) {
            int to_swap = local_index ^ cur_len;

            if (to_swap > local_index) {
                uint a = loc_mem[local_index];
                uint b = loc_mem[to_swap];
                if (((local_index & cur_block) == 0) == (a > b)) {
                    loc_mem[local_index] = b;
                    loc_mem[to_swap] = a;
                }
            }

            cur_len >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        cur_block += cur_block;
    }

    output_data[global_index] = loc_mem[local_index];
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  pw2, // new blocks will have size (1 << pw2)
                   int  n)
{
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int global_block_start = (global_index & (~((1 << pw2) - 1)));

    if (global_block_start + (1 << (pw2 - 1)) >= n) {
        output_data[global_index] = input_data[global_index];
        return;
    }

    int local_block_start = (global_index & (~(GROUP_SIZE - 1)));
    int flex = local_block_start - global_block_start + 1;
    int right_size = min(1 << (pw2 - 1), n - global_block_start - (1 << (pw2 - 1)));

    __local first_left;

    // flex - m <= right_size
    // flex - right_size <= m

    int l = max(0, flex - right_size);
    int r = min((1 << (pw2 - 1)), flex) + 1;

    if (local_index == 0) {
        while (r - l > 1) {
            int m = (l + r) / 2;

            if (input_data[global_block_start + m - 1] <= input_data[global_block_start + (1 << (pw2 - 1)) + flex - m]) {
                l = m;
            } else {
                r = m;
            }
        }
        first_left = l;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint left_part[GROUP_SIZE];
    __local uint right_part[GROUP_SIZE];

    int left_part_start_from_global = global_block_start + max(first_left - 1, 0);

    left_part[local_index] = (left_part_start_from_global + local_index < global_block_start + (1 << (pw2 - 1)) ? input_data[left_part_start_from_global + local_index] : UINT_MAX);

    int right_part_start_from_global = global_block_start + (1 << (pw2 - 1)) + max(0, flex - first_left - 1);

    right_part[local_index] = (right_part_start_from_global + local_index < global_block_start + (1 << (pw2 - 1)) + right_size ? input_data[right_part_start_from_global + local_index] : UINT_MAX);

    barrier(CLK_LOCAL_MEM_FENCE);

    flex = global_index - global_block_start + 1;

    l = max(first_left, flex - right_size);
    r = min(min(flex, 1 << (pw2 - 1)), first_left + local_index) + 1;

    while (r - l > 1) {
        int m = (l + r) / 2;

        int left_index = global_block_start + m - 1;
        int right_index = global_block_start + (1 << (pw2 - 1)) + flex - m;

        left_index -= left_part_start_from_global;
        right_index -= right_part_start_from_global;

        if (left_part[left_index] <= right_part[right_index]) {
            l = m;
        } else {
            r = m;
        }
    }

    int left_index = global_block_start + l - 1;
    int right_index = global_block_start + (1 << (pw2 - 1)) + flex - l - 1;

    uint cand_l = (left_index >= left_part_start_from_global ? left_part[left_index - left_part_start_from_global] : 0);
    uint cand_r = (right_index >= right_part_start_from_global ? right_part[right_index - right_part_start_from_global] : 0);

    uint res = max(cand_l, cand_r);
    output_data[global_index] = res;
}