#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* offsets,
    __global       uint* output,
    uint n,
    uint shift)
{
    uint lane = get_local_id(0);
    uint index = get_global_id(0);
    uint value = index < n ? values[index] : 0;
    uint bucket = (value >> shift) & (RADIX_BUCKETS - 1);
    __local uint buckets[GROUP_SIZE];
    buckets[lane] = bucket;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (index < n) {
        uint local_offset = 0;
        for (uint previous = 0; previous < lane; ++previous) {
            if (buckets[previous] == bucket) {
                ++local_offset;
            }
        }
        uint global_offset = offsets[bucket * get_num_groups(0) + get_group_id(0)];
        output[global_offset + local_offset] = value;
    }
}
