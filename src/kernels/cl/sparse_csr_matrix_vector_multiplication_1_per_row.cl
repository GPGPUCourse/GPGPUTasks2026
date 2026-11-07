#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_1_per_row(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint nnz,
    const uint rows
) {
    const uint gid = get_global_id(0);

    if (gid >= rows) return;

    const uint row_offset = csr_row_offsets[gid];
    const uint row_size = gid < (rows - 1) ? csr_row_offsets[gid + 1] - csr_row_offsets[gid] : nnz - row_offset;

    output_vector_values[gid] = 0;
    for (uint i = 0; i < row_size; ++i) {
        const uint vv = vector_values[csr_columns[row_offset + i]];
        const uint vm = csr_values[row_offset + i];
        output_vector_values[gid] += vm * vv;
    }
}
