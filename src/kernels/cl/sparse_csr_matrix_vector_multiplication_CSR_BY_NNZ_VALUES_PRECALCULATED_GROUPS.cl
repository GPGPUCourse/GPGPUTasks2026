#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Каждый поток обрабатывает до SMALL_GROUP_SIZE последовательных NNZ.
__attribute__((reqd_work_group_size(GROUP_SIZE / SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_CSR_BY_NNZ_VALUES_PRECALCULATED_GROUPS(
    __global const uint* row_offsets, // size=nrows+1
    __global const uint* rows, // size=nnz
    __global const uint* col, // size=nnz
    __global const uint* val, // size=nnz
    __global const uint* vector, // size=ncols
          __global uint* output, // size=nrows
                    uint nnz
)
{
    const uint global_id = get_global_id(0);
    const uint i = global_id * SMALL_GROUP_SIZE;
    const uint local_begin = get_local_id(0) * SMALL_GROUP_SIZE;

    if (i >= nnz) {
        return;
    }

    __local uint val_buffer[GROUP_SIZE];
    __local uint col_buffer[GROUP_SIZE];
    __local uint row_buffer[GROUP_SIZE];
    const uint count = min((uint)SMALL_GROUP_SIZE, nnz - i);
    for (uint j = 0; j < count; ++j) {
        val_buffer[local_begin + j] = val[i + j];
        col_buffer[local_begin + j] = col[i + j];
        row_buffer[local_begin + j] = rows[i + j];
    }

    uint last_row = row_buffer[local_begin];
    uint current_sum = 0;
    for (uint j = 0; j < count; ++j) {
        if (row_buffer[local_begin + j] > last_row) {
            atomic_add(&output[last_row], current_sum);
            current_sum = 0;
            last_row = row_buffer[local_begin + j];
        }
        current_sum += val_buffer[local_begin + j] * vector[col_buffer[local_begin + j]];
    }
    if (current_sum > 0) {
        atomic_add(&output[last_row], current_sum);
    }
}
