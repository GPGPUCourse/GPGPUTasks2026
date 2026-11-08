#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(__global uint* hist, uint m)
{
    const uint lid = get_local_id(0);
    __local uint s[GROUP_SIZE];
    uint carry = 0u;

    for (uint base = 0; base < m; base += GROUP_SIZE) {
        const uint idx = base + lid;
        const uint v = idx < m ? hist[idx] : 0u;
        s[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint off = 1; off < GROUP_SIZE; off <<= 1) {
            const uint t = lid >= off ? s[lid - off] : 0u;
            barrier(CLK_LOCAL_MEM_FENCE);
            s[lid] += t;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (idx < m)
            hist[idx] = carry + s[lid] - v; // exclusive
        carry += s[GROUP_SIZE - 1u];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}