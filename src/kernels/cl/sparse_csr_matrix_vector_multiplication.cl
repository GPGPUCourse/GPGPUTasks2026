#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define VT    SPMV_VALUES_PER_THREAD
#define BLOCK SPMV_BLOCK_SIZE

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
#define PADDED(i) ((i) + CONFLICT_FREE_OFFSET(i))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
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
    __local uint block_products[PADDED(BLOCK - 1) + 1];

    const unsigned int local_id = get_local_id(0);
    const unsigned int block_start = get_group_id(0) * BLOCK;

    for (unsigned int k = 0; k < VT; ++k) {
        const unsigned int j = k * GROUP_SIZE + local_id;
        const unsigned int i = block_start + j;
        block_products[PADDED(j)] = (i < nnz) ? values[i] * vector_values[columns[i]] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int from = block_start + local_id * VT;
    if (from >= nnz)
        return;
    const unsigned int to = min(from + VT, nnz);

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
            sum += block_products[PADDED(i - block_start)];
        atomic_add(&output[row], sum);
    }
}
