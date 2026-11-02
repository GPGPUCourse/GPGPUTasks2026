#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

unsigned int get_group_diag(
    __global const uint* input_data,
                   int  iter,
                   int  n,
                   int  gid)
{
    const unsigned int base = (gid * GROUP_SIZE) & ~((1 << (iter + 1)) - 1);
    const unsigned int d = (gid * GROUP_SIZE) & ((1 << (iter + 1)) - 1);
    const int shift = (1 << iter);
    if (gid * GROUP_SIZE >= n) {
        return 0;
    }

    const unsigned int block_len = MIN(1 << (iter + 1), (unsigned)n - base);
    const unsigned int left_cnt = MIN(1 << iter, block_len);
    const unsigned int right_cnt = block_len - left_cnt;

    unsigned int l = MAX(0, (int)d - (int)right_cnt) - 1, r = MIN(d, left_cnt) + 1;
    while (l + 1 < r) {
        unsigned int m = (l + r) / 2;
        if (m == 0 || d - m >= right_cnt || input_data[base + m - 1] <= input_data[base + shift + d - m]) {
            l = m;
        } else {
            r = m;
        }
    }

    return l;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  iter,
                   int  n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    
    const unsigned int shift = (1 << iter);
    const unsigned int block_mask = (1 << (iter + 1)) - 1;

    const unsigned int base = gid & ~block_mask;

    const unsigned int gr_id = get_group_id(0);
    const unsigned int gr_d = (gr_id * GROUP_SIZE) & block_mask;

    const unsigned int block_len = MIN(1 << (iter + 1), (unsigned)n - base);
    const unsigned int block_width = MIN(GROUP_SIZE, (unsigned)n - gr_id * GROUP_SIZE);

    const bool left_cnt_cond = gr_d + block_width >= block_len;

    __local unsigned int group_diag_loc, group_diag_next_loc;
    if (lid == 0) {
        group_diag_loc = get_group_diag(input_data, iter, n, gr_id);
    } else if (lid == 32 && !left_cnt_cond) {
        group_diag_next_loc = get_group_diag(input_data, iter, n, gr_id + 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int group_diag = group_diag_loc, group_diag_next = group_diag_next_loc;

    const unsigned int left_cnt_prev = group_diag;
    const unsigned int right_cnt_prev = gr_d == 0 ? 0 : gr_d - left_cnt_prev;

    const unsigned int left_cnt = (left_cnt_cond ? MIN(1 << iter, block_len) : group_diag_next) - group_diag_loc;
    const unsigned int right_cnt = block_width - left_cnt;
    

    __local unsigned int buf[GROUP_SIZE];
    unsigned int idx;
    if (lid < left_cnt) {
        idx = base + lid + left_cnt_prev;
    } else {
        idx = base + shift + right_cnt_prev + (lid - left_cnt);
    }
    buf[lid] = input_data[idx];

    barrier(CLK_LOCAL_MEM_FENCE);

    int l = -1, r = MIN(lid, left_cnt);
    while (l + 1 < r) {
        unsigned int m = (l + r) / 2;

        const bool lim_r = lid - m < right_cnt;

        if (lim_r && buf[m] > buf[left_cnt + lid - m]) {
            r = m;
        } else {
            l = m;
        }
    }

    {
        const bool lim_l = r < left_cnt, lim_r = lid - r < right_cnt;
        if (!((!lim_l && lim_r) || (lim_l && lim_r && buf[r] > buf[left_cnt + lid - r]))) {
            l = r;
        }
        
    }

    unsigned int this = (l >= left_cnt ? 0 : buf[l]);
    unsigned int other = (lid - l - 1 >= right_cnt ? 0 : buf[left_cnt + lid - l - 1]);
    output_data[gid] = MAX(this, other);
}
