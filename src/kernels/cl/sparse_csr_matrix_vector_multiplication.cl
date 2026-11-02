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
         __global       uint* output_values,
         uint n_rows)
{
    const uint row = get_group_id(0);
    if (row >= n_rows) return;

    const uint local_id = get_local_id(0);
    uint offset = local_id;
    uint local_sum = 0;

    uint row_begin = csr_row_offsets[row];
    uint row_end   = csr_row_offsets[row + 1];
    for (uint i = row_begin + offset; i < row_end; i += GROUP_SIZE){
        local_sum += csr_values[i] * vector_values[csr_columns[i]];
    }

    __local uint partial[GROUP_SIZE];
    partial[offset] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sum = 0;

    if (local_id == 0){
        for (uint i = 0; i < GROUP_SIZE; ++i){
            sum += partial[i];
        }
        output_values[row] = sum;
    }


}
