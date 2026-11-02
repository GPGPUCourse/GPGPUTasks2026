#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define GET(x) \
    input_data[((x) < left_cnt) ? base + (x) + left_cnt_prev : base + shift + right_cnt_prev + ((x) - left_cnt)]

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global const uint* group_diags,
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

    const unsigned int left_cnt =
        (gr_d + block_width >= block_len ? MIN(1 << iter, block_len) : group_diags[gr_id + 1]) - group_diags[gr_id];
    const unsigned int right_cnt = block_width - left_cnt;
    
    const unsigned int left_cnt_prev = group_diags[gr_id];
    const unsigned int right_cnt_prev =
        ((gr_id * GROUP_SIZE) & block_mask) == 0
        ? 0 : gr_d - left_cnt_prev;

    int l = -1, r = MIN(lid, left_cnt);

    unsigned int this, other;

    while (l + 1 < r) {
        unsigned int m = (l + r) / 2;

        const bool lim_r = lid - m < right_cnt;

        if (lim_r && GET(m) > GET(left_cnt + lid - m)) {
            r = m;
        } else {
            l = m;
        }
    }

    const bool lim_l = r < left_cnt, lim_r = lid - r < right_cnt;
    if (!((!lim_l && lim_r) || (lim_l && lim_r && GET(r) > GET(left_cnt + lid - r)))) {
        l = r;
    }
    this = (l >= left_cnt ? 0 : GET(l));
    other = (lid - l - 1 >= right_cnt ? 0 : GET(left_cnt + lid - l - 1));
        
    output_data[gid] = MAX(this, other);
}
