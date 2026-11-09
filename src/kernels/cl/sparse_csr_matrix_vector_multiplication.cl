#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* cols,
    __global const uint* vals,
    __global const uint* vec,
    __global       uint* res,
    unsigned int nrows)
{
    unsigned int row = get_global_id(0);
    if (row >= nrows) {
        return;
    }

    unsigned int start = row_offsets[row];
    unsigned int end = row_offsets[row + 1];

    unsigned int sum = 0;
    for (uint i = start; i < end; ++i) {
        sum += vals[i] * vec[cols[i]];
    }

    res[row] = sum;
}
