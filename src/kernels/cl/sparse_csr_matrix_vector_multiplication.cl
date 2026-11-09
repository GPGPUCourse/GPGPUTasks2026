#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* rows,
    __global const uint* cols,
    __global const uint* vals,
    __global const uint* x,
    __global uint* y,
    unsigned int nnz)
{
    int i = get_global_id(0);
    if (i >= nnz)
        return;

    atomic_add(&y[rows[i]], vals[i] * x[cols[i]]);
}
