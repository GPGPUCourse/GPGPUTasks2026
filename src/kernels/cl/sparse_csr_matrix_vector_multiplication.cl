#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define ROW_LANES SMALL_GROUP_SIZE
#define ROWS_PER_GROUP (GROUP_SIZE / ROW_LANES)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector_values,
                   uint  nrows)
{
    const uint local_id = get_local_id(0);
    const uint lane = local_id % ROW_LANES;
    const uint row_in_group = local_id / ROW_LANES;
    const uint row = get_group_id(0) * ROWS_PER_GROUP + row_in_group;

    __local uint partial_sums[GROUP_SIZE];

    uint acc = 0;
    if (row < nrows) {
        const uint row_from = csr_row_offsets[row];
        const uint row_to = csr_row_offsets[row + 1];
        for (uint i = row_from + lane; i < row_to; i += ROW_LANES) {
            acc += csr_values[i] * vector_values[csr_columns[i]];
        }
    }
    partial_sums[local_id] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = ROW_LANES / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial_sums[local_id] += partial_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0 && row < nrows) {
        output_vector_values[row] = partial_sums[local_id];
    }
}
