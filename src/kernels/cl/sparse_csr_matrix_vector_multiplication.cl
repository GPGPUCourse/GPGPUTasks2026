#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SMALL_GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global       uint* result)
{
    __local uint sums[SMALL_GROUP_SIZE];
    unsigned int tid = get_local_id(0);
    unsigned int row = get_group_id(0);

    uint sum = 0;
    for (unsigned int i = row_offsets[row] + tid; i < row_offsets[row + 1]; i += SMALL_GROUP_SIZE) {
        sum += values[i] * vector_values[columns[i]];
    }
    sums[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = SMALL_GROUP_SIZE / 2; step > 0; step /= 2) {
        if (tid < step) {
            sums[tid] += sums[tid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        result[row] = sums[0];
    }
}
