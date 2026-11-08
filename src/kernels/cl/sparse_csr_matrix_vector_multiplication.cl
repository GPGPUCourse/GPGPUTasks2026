#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_values,
    uint nrows)
{
    const uint local_index = get_local_id(0);
    const uint lane = local_index % SMALL_GROUP_SIZE;
    const uint row_in_group = local_index / SMALL_GROUP_SIZE;
    const uint rows_per_group = GROUP_SIZE / SMALL_GROUP_SIZE;
    const uint row = get_group_id(0) * rows_per_group + row_in_group;

    uint sum = 0u;
    if (row < nrows) {
        const uint row_begin = csr_row_offsets[row];
        const uint row_end = csr_row_offsets[row + 1u];
        for (uint index = row_begin + lane; index < row_end; index += SMALL_GROUP_SIZE) {
            sum += csr_values[index] * vector_values[csr_columns[index]];
        }
    }

    __local uint partial_sums[GROUP_SIZE];
    partial_sums[local_index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            partial_sums[local_index] += partial_sums[local_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0u && row < nrows) {
        output_values[row] = partial_sums[local_index];
    }
}
