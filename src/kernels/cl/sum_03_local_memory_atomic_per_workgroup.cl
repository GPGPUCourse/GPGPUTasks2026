#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const uint index = get_global_id(0) * SUM_03_VALUES_PER_ITEM;
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    uint value = 0;

    if (index + SUM_03_VALUES_PER_ITEM <= n) {
        for (uint i = 0; i < SUM_03_VALUES_PER_ITEM; i += 4) {
            const uint4 values = vload4(0, a + index + i);
            value += values.s0 + values.s1 + values.s2 + values.s3;
        }
    } else {
        for (uint i = 0; i < SUM_03_VALUES_PER_ITEM; ++i) {
            if (index + i < n) {
                value += a[index + i];
            }
        }
    }
    local_data[local_index] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
        if (local_index < offset){
            local_data[local_index] += local_data[local_index + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        atomic_add(sum, local_data[0]);
    }
}
