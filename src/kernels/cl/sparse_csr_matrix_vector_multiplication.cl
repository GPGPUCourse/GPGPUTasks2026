#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_with_zeros(
    __global uint* output,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
        output[i] = 0;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global       uint* output,
    unsigned int nrows,
    unsigned int nnz)
{
    const unsigned int from = get_global_id(0) * SPMV_VALUES_PER_ITEM;
    if (from >= nnz)
        return;
    const unsigned int to = min(from + SPMV_VALUES_PER_ITEM, nnz);

    unsigned int lo = 0, hi = nrows;
    while (hi - lo > 1) {
        const unsigned int mid = (lo + hi) / 2;
        if (row_offsets[mid] <= from)
            lo = mid;
        else
            hi = mid;
    }

    unsigned int i = from;
    for (unsigned int row = lo; i < to; ++row) {
        const unsigned int row_to = min(row_offsets[row + 1], to);
        uint sum = 0;
        for (; i < row_to; ++i)
            sum += values[i] * vector_values[columns[i]];
        atomic_add(&output[row], sum);
    }
}
