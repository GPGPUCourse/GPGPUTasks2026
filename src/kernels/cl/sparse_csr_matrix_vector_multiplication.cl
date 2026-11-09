#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    uint const __global csr_row_offsets,
    uint const __global csr_values,
    uint __global output,
    uint nnz,
    uint rows
{
    uint global_i = get_global_id(0);
    uint sum = 0;
    uint offset = csr_row_offsets[global_i];
    if (global_i < rows-1) {
        uint next = csr_row_offsets[global_i + 1] - offset;
    } else {
        uint next = nnz - offset;
    }
    for (int i = 0; i < next; i++) {
        sum += csr_values[global_i + i];
    }
    output[global_i] = sum;
}
