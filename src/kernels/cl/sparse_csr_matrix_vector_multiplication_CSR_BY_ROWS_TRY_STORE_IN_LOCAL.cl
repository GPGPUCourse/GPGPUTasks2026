#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// кернел запускается от каждой строки матрицы (их nrows)
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication_CSR_BY_ROWS_TRY_STORE_IN_LOCAL(
    __global const uint* row_offsets, // size=nrows+1
    __global const uint* col, // size=nnz
    __global const uint* val, // size=nnz
    __global const uint* vector, // size=ncols
          __global uint* output, // size=nrows
                    uint nrows
)
{
    const uint i = get_global_id(0);
    const uint local_i = get_local_id(0);
    // __local uint val_buffer[GROUP_SIZE][32];
    __local uint col_buffer[GROUP_SIZE][32];

    if (i >= nrows) {
        return;
    }
    
    const uint l = row_offsets[i];
    const uint r = row_offsets[i + 1];

    uint result = 0;
    for (uint block_begin = l; block_begin < r;) {
        const uint count = min(32u, r - block_begin);
        for (uint j = 0; j < count; ++j) {
            // val_buffer[local_i][j] = val[block_begin + j];
            col_buffer[local_i][j] = col[block_begin + j];
        }

        for (uint j = 0; j < count; ++j) {
            result += val[block_begin + j] * vector[col_buffer[local_i][j]];
        }
        block_begin += count;
    }

    output[i] = result;
}
