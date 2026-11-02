#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_group_diag(
    __global const uint* input_data,
    __global       uint* group_diags,
                   int  iter,
                   int  n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int base = (gid * GROUP_SIZE) & ~((1 << (iter + 1)) - 1);
    const unsigned int d = (gid * GROUP_SIZE) & ((1 << (iter + 1)) - 1);
    const int shift = (1 << iter);
    if (gid * GROUP_SIZE >= n) {
        return;
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

    group_diags[gid] = l;
}
