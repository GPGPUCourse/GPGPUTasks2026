#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(SCAN_GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* block_sums,
    unsigned int n,
    unsigned int input_offset,
    unsigned int output_offset)
{
    input += input_offset;
    block_sums += output_offset;
    const uint lid = get_local_id(0);
    const size_t base = get_group_id(0) * SCAN_BLOCK_SIZE + 4 * lid;
    uint4 values = (uint4)(0);
    if (base + 3 < n) {
        values = vload4(0, input + base);
    } else {
        for (uint i = 0; i < 4; ++i)
            if (base + i < n)
                values[i] = input[base + i];
    }

    __local uint sums[SCAN_GROUP_SIZE];
    sums[lid] = values.x + values.y + values.z + values.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SCAN_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            sums[lid] += sums[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        block_sums[get_group_id(0)] = sums[0];
}
