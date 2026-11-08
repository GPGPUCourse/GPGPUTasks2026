#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define ROWS_PER_GROUP (GROUP_SIZE / SMALL_GROUP_SIZE)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector_values,
                   uint  nrows)
{
    __local uint sums[GROUP_SIZE];

    const uint lid = get_local_id(0);
    const uint lane = lid % SMALL_GROUP_SIZE;
    const uint row = get_group_id(0) * ROWS_PER_GROUP + lid / SMALL_GROUP_SIZE;

    uint acc = 0;
    if (row < nrows) {
        const uint from = csr_row_offsets[row];
        const uint to = csr_row_offsets[row + 1];
        for (uint i = from + lane; i < to; i += SMALL_GROUP_SIZE) {
            acc += csr_values[i] * vector_values[csr_columns[i]];
        }
    }

    sums[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            sums[lid] += sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < nrows && lane == 0) {
        output_vector_values[row] = sums[lid];
    }
}
