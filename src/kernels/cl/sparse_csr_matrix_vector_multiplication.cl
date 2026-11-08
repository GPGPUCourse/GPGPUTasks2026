#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// SpMV для CSR: y = A * x
// Одна work-group размера SMALL_GROUP_SIZE (warp) обрабатывает одну строку.
// Нити группы параллельно бегут по ненулевым элементам строки с шагом SMALL_GROUP_SIZE,
// затем складывают частичные суммы в local memory.
__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets, // nrows + 1
    __global const uint* csr_columns,     // nnz
    __global const uint* csr_values,      // nnz
    __global const uint* vector_values,   // ncols
    __global       uint* output_values,   // nrows
                   uint  nrows)
{
    const uint row = get_group_id(0);
    const uint lane = get_local_id(0);
    if (row >= nrows)
        return;

    const uint row_begin = csr_row_offsets[row];
    const uint row_end = csr_row_offsets[row + 1];

    // ulong, как size_t на CPU: промежуточная сумма не должна обрезаться до 32 бит.
    ulong sum = 0;
    for (uint i = row_begin + lane; i < row_end; i += SMALL_GROUP_SIZE)
        sum += (ulong) csr_values[i] * (ulong) vector_values[csr_columns[i]];

    __local ulong partial[SMALL_GROUP_SIZE];
    partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0)
        output_values[row] = (uint) partial[0];
}
