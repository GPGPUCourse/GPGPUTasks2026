#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  iter,
                   int  n)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = x ^ (1 << iter);
    if (x >= n) {
        return;
    }

    const unsigned int vx = input_data[x];

    const bool on_right = (x >> iter) & 1;
    const unsigned int base = x & ~((1 << (iter + 1)) - 1);
    const unsigned int this_base = x & ~((1 << iter) - 1);
    const unsigned int other_base = y & ~((1 << iter) - 1);

    unsigned int l = 0, r = (1 << iter);
    while (l + 1 < r) {
        const unsigned int m = (l + r) / 2;
        const unsigned int vm = (other_base + m < n ? input_data[other_base + m] : 0xFFFFFFFF);
        if (vm < vx || (on_right && vm == vx)) {
            l = m;
        } else {
            r = m;
        }
    }

    const unsigned int vl = (other_base + l < n ? input_data[other_base + l] : 0xFFFFFFFF);
    const unsigned int found = (vl < vx || (on_right && vl == vx)) ? r : l;
    const unsigned int idx = found + (x - this_base) + base;

    output_data[idx] = vx;
}
