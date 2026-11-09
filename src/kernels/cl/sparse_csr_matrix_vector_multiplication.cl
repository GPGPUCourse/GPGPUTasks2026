#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values)
{
    const uint row = get_group_id(0);
    const uint lane = get_local_id(0);
    const uint end = csr_row_offsets[row + 1];
    uint sum = 0;
    for (uint i = csr_row_offsets[row] + lane; i < end; i += SMALL_GROUP_SIZE)
        sum += csr_values[i] * vector_values[csr_columns[i]];

    __local uint sums[SMALL_GROUP_SIZE];
    sums[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride)
            sums[lane] += sums[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0)
        output_vector_values[row] = sums[0];
}
