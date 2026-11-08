#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* csr_values,
    __global const uint* vector,
    __global       uint* output,
             const uint  nrows
) {
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint row_id = local_id % SMALL_GROUP_SIZE;
    const uint row = group_id * ROWS_PER_GROUP + local_id / SMALL_GROUP_SIZE;
    __local uint accums[GROUP_SIZE];

    uint accum = 0;

    if (row < nrows) {
        const uint row_begin = row_offsets[row];
        const uint row_end = row_offsets[row + 1];
        for (uint index = row_begin + row_id; index < row_end; index += SMALL_GROUP_SIZE)
            accum += csr_values[index] * vector[columns[index]];
    }

    accums[local_id] = accum;

    for (uint shift = 1; shift < SMALL_GROUP_SIZE; shift *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row_id * shift * 2 < SMALL_GROUP_SIZE)
            accums[local_id + row_id * (shift * 2 - 1)] += accums[local_id + row_id * (shift * 2 - 1) + shift];
    }

    if (row_id == 0 && row < nrows)
        output[row] = accums[local_id];
}
