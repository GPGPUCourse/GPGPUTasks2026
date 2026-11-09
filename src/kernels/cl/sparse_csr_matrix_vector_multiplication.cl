#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global uint* output,
    unsigned int nrows)
{
    const uint lid = get_local_id(0);
    const uint lane = lid % SMALL_GROUP_SIZE;
    const size_t row = get_global_id(0) / SMALL_GROUP_SIZE;
    uint sum = 0;
    if (row < nrows) {
        const uint end = row_offsets[row + 1];
        for (size_t i = (size_t)row_offsets[row] + lane; i < end; i += SMALL_GROUP_SIZE) {
            sum += values[i] * vector_values[columns[i]];
        }
    }

    __local uint partial_sums[GROUP_SIZE];
    partial_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial_sums[lid] += partial_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < nrows) {
        output[row] = partial_sums[lid];
    }
}
