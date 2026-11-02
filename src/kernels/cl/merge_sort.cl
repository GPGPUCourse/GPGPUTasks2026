#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define GUARDL(x, v) \
    ((x) < group_diag ? (v) : 0)
#define GUARDR(x, v) \
    ((group_diag <= (x) && (x) < GROUP_SIZE) ? (v) : 0x8FFFFFFF)
#define GETL(x) GUARDL(x, buf[x])
#define GETR(x) GUARDR(x, buf[x])

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))


unsigned int get_group_diag(
    __global const uint* input_data,
    int iter, int n, int gid)
{
    const unsigned int base = (gid * GROUP_SIZE) & ~((1 << (iter + 1)) - 1);
    const unsigned int d = (gid * GROUP_SIZE) & ((1 << (iter + 1)) - 1);
    const int shift = (1 << iter);
    if (gid * GROUP_SIZE >= n) {
        return 9;
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
    const unsigned int base = gid & ~((1 << (iter + 1)) - 1);
    const unsigned int shift = (1 << iter);

    const unsigned int gr_id = get_group_id(0);
    const unsigned int gr_d = (gr_id * GROUP_SIZE) & ((1 << (iter + 1)) - 1);

    const unsigned int block_len = MIN(1 << (iter + 1), (unsigned)n - base);
    const unsigned int block_width = MIN(GROUP_SIZE, (unsigned)n - gr_id * GROUP_SIZE);

    const unsigned int gr_diag_next = get_group_diag(input_data, iter, n, gr_id + 1);
    const unsigned int gr_diag = get_group_diag(input_data, iter, n, gr_id);

    const unsigned int left_cnt =
        (gr_d + block_width >= block_len || ((((gr_id + 1) * GROUP_SIZE) & ((1 << (iter + 1)) - 1)) == 0)
        ? MIN(1 << iter, block_len) : gr_diag_next) - gr_diag;
    const unsigned int right_cnt = block_width - left_cnt;
    
    const unsigned int left_cnt_prev = gr_diag;
    const unsigned int right_cnt_prev =
        ((gr_id * GROUP_SIZE) & ((1 << (iter + 1)) - 1)) == 0
        ? 0 : gr_d - left_cnt_prev;

    __local unsigned int buf[GROUP_SIZE];
    if (lid < block_width) {
        if (lid < left_cnt) {
            buf[lid] = input_data[base + lid + left_cnt_prev];
        } else {
            buf[lid] = input_data[base + shift + right_cnt_prev + (lid - left_cnt)];
        }
    } else {
        buf[lid] = 0xFFFFFFFFu;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int bar = MIN(lid, left_cnt) + 1;

    unsigned int l = -1, r = bar;
    while (l + 1 < r) {
        unsigned int m = (l + r) / 2;

        const unsigned int vl = (m < left_cnt) ? buf[m] : 0xFFFFFFFFu;
        const unsigned int vr = (lid - m < right_cnt) ? buf[left_cnt + lid - m] : 0xFFFFFFFFu;
        if (vl <= vr) {
            l = m;
        } else {
            r = m;
        }
    }

    
    unsigned int this = ((l < 0 || l >= left_cnt) ? 0 : buf[l]);
    unsigned int other = ((lid - l - 1 < 0 || lid - l - 1 >= right_cnt) ? 0 : buf[left_cnt + lid - l - 1]);
    if (lid < block_width) {
        output_data[gid] = MAX(this, other);
    }
}
