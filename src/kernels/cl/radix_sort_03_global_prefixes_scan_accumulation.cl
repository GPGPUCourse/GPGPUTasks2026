#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* input,
    __global const uint* block_prefixes,
    __global uint* output,
    unsigned int n,
    unsigned int data_offset,
    unsigned int prefixes_offset)
{
    input += data_offset;
    output += data_offset;
    const uint lid = get_local_id(0);
    const size_t block = get_group_id(0);
    const size_t base = block * SCAN_BLOCK_SIZE + 4 * lid;
    uint4 values = (uint4)(0);
    if (base + 3 < n) {
        values = vload4(0, input + base);
    }
    else {
        for (uint i = 0; i < 4; ++i) {
            if (base + i < n) {
                values[i] = input[base + i];
            }
        }
    }
    __local uint sums[GROUP_SIZE];
    sums[lid] = values.x + values.y + values.z + values.w;

    values = (uint4)(0, values.x, values.x + values.y, values.x + values.y + values.z);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        const uint index = (lid + 1) * 2 * stride - 1;
        if (index < GROUP_SIZE) {
            sums[index] += sums[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        sums[GROUP_SIZE - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        const uint index = (lid + 1) * 2 * stride - 1;
        if (index < GROUP_SIZE) {
            const uint left = sums[index - stride];
            sums[index - stride] = sums[index];
            sums[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint offset = sums[lid] + (get_num_groups(0) == 1
        ? 0 : block_prefixes[prefixes_offset + block]);
    values += (uint4)(offset);
    if (base + 3 < n) {
        vstore4(values, 0, output + base);
    }
    else {
        for (uint i = 0; i < 4; ++i)
            if (base + i < n)
                output[base + i] = values[i];
    }
}
