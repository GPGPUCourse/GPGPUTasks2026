#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define GET(x) \
    (iter == 0 ? input_data[group_base + x] : ((iter & 1) ? buf1[x] : buf2[x]))
#define PUT(x, v) \
    (iter == GROUP_SHIFT - 1 ? (output_data[group_base + x] = (v)) : (!(iter & 1) ? (buf1[x] = (v)) : (buf2[x] = (v))))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_one_group(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int x = get_local_id(0);
    const unsigned int group_base = get_group_id(0) * GROUP_SIZE;
    __local unsigned int buf1[GROUP_SIZE];
    __local unsigned int buf2[GROUP_SIZE];

    for (int iter = 0; iter < GROUP_SHIFT; ++iter) {
        const unsigned int y = x ^ (1 << iter);

        const unsigned int vx = GET(x);

        const bool on_right = (x >> iter) & 1;
        const unsigned int base = x & ~((1 << (iter + 1)) - 1);
        const unsigned int this_base = x & ~((1 << iter) - 1);
        const unsigned int other_base = y & ~((1 << iter) - 1);

        unsigned int l = 0, r = (1 << iter);
        while (l + 1 < r) {
            const unsigned int m = (l + r) / 2;
            const unsigned int vm = GET(other_base + m);
            if (vm < vx || (on_right && vm == vx)) {
                l = m;
            } else {
                r = m;
            }
        }

        const unsigned int vl = GET(other_base + l);
        const unsigned int found = (vl < vx || (on_right && vl == vx)) ? r : l;
        const unsigned int idx = found + (x - this_base) + base;

        PUT(idx, vx);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#undef GET
#undef PUT
