#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_1_per_cell_atomic(
    __global const uint* csr_rows,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint nnz
) {
    const uint gid = get_global_id(0);

    if (gid >= nnz) return;

    const uint row = csr_rows[gid];

    const uint vv = vector_values[csr_columns[gid]];
    const uint vm = csr_values[gid];
    atomic_add(output_vector_values + row, vm * vv);
}