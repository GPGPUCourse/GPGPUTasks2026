#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(__global const uint*   csr_row_offsets,
                                                      __global const uint*       csr_columns,
                                                      __global const uint*        csr_values,
                                                      __global const uint*     vector_values,
                                                      __global       uint* output_csr_values,
                                                                            unsigned int nrows)
{
    const uint row = get_group_id(0);
    const uint lane = get_local_id(0);

    if (row >= nrows) {
        return;
    }

    const uint row_from = csr_row_offsets[row];
    const uint row_to = csr_row_offsets[row + 1];

    uint accumulator = 0;

    for (uint i = row_from + lane; i < row_to; i += SMALL_GROUP_SIZE) {
        const uint col = csr_columns[i];
        accumulator += csr_values[i] * vector_values[col];
    }

    __local uint sums[SMALL_GROUP_SIZE];

    sums[lane] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = SMALL_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sums[lane] += sums[lane + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
        output_csr_values[row] = sums[0];
    }
}
