#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                      __global       uint* b,
                                             unsigned int  n)
{
    const uint index = get_global_id(0) * SUM_04_VALUES_PER_ITEM;
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);

    __local uint local_data[GROUP_SIZE];

    uint value = 0;
    if (index + SUM_04_VALUES_PER_ITEM <= n) {
        for (uint i = 0; i < SUM_04_VALUES_PER_ITEM; i += 4) {
            const uint4 values = vload4(0, a + index + i);
            value += values.s0 + values.s1 + values.s2 + values.s3;
        }
    } else {
        for (uint i = 0; i < SUM_04_VALUES_PER_ITEM; ++i) {
            if (index + i < n)
                value += a[index + i];
        }
    }
    local_data[lid] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
        if (lid < offset) {
            local_data[lid] += local_data[lid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        b[group] = local_data[0];
    }
}
