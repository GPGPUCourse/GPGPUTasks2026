#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets_gpu,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector,
    __global uint* output,
    const unsigned int n
)
{
    const unsigned int ind = get_global_id(0);
    if (ind >= n) {
        return;
    }

    unsigned int sum = 0;
    for (unsigned int i = csr_row_offsets_gpu[ind]; i < csr_row_offsets_gpu[ind + 1]; i++) {
        sum += csr_values[i] * vector[csr_columns[i]];
    }
    output[ind] = sum;
}
