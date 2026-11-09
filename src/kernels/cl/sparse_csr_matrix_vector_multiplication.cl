#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* column_indices,
    __global const uint* matrix_values,
    __global const uint* vector_values,
    __global uint* result_vector,
    uint nrows)
{
    const size_t row = get_group_id(0);
    const uint lane = get_local_id(0);
    if (row >= nrows)
        return;

    const uint row_from = row_offsets[row];
    const uint row_to = row_offsets[row + 1];
    uint acc = 0;
    for (size_t i = (size_t)row_from + lane; i < row_to; i += SMALL_GROUP_SIZE) {
        acc += matrix_values[i] * vector_values[column_indices[i]];
    }

    __local uint partial_sums[SMALL_GROUP_SIZE];
    partial_sums[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial_sums[lane] += partial_sums[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        result_vector[row] = partial_sums[0];
    }
}
