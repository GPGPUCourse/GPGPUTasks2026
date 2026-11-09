#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global uint* output,
    uint nrows)
{
    const uint row = get_group_id(0);
    const uint tid = get_local_id(0);

    if (row >= nrows)
        return;

    uint sum = 0;
    const uint end = row_offsets[row + 1];
    for (uint j = row_offsets[row] + tid; j < end; j += SMALL_GROUP_SIZE)
        sum += values[j] * vector_values[columns[j]];

    __local uint partial_sums[SMALL_GROUP_SIZE];
    partial_sums[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            partial_sums[tid] += partial_sums[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        output[row] = partial_sums[0];
}
