#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector_values,
    uint nrows)
{
    const uint row = get_global_id(1);
    const uint lane = get_local_id(0);
    const uint local_row = get_local_id(1);

    uint sum = 0;
    if (row < nrows) {
        for (uint i = csr_row_offsets[row] + lane; i < csr_row_offsets[row + 1]; i += GROUP_SIZE_X) {
            sum += csr_values[i] * vector_values[csr_columns[i]];
        }
    }

    __local uint partial[GROUP_SIZE_Y][GROUP_SIZE_X];

    partial[local_row][lane] = sum;
    for (uint stride = GROUP_SIZE_X / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < stride)
            partial[local_row][lane] += partial[local_row][lane + stride];
    }

    if (lane == 0 && row < nrows)
        output_vector_values[row] = partial[local_row][0];
}
