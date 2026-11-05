#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication( // TODO input/output buffers
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector_values,
                   uint nrows)
{
    // TODO
    const uint row = get_global_id(0) / SMALL_GROUP_SIZE;
    const uint local_id = get_local_id(0);
    const uint lane = local_id % SMALL_GROUP_SIZE;
    uint sum = 0;
    if (row < nrows) {
        const uint end = csr_row_offsets[row + 1];
        for (uint i = csr_row_offsets[row] + lane; i < end; i += SMALL_GROUP_SIZE) {
            sum += csr_values[i] * vector_values[csr_columns[i]];
        }
    }

    __local uint sums[GROUP_SIZE];
    sums[local_id] = sum;
    for (uint stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < stride) {
            sums[local_id] += sums[local_id + stride];
        }
    }
    if (lane == 0 && row < nrows) {
        output_vector_values[row] = sums[local_id];
    }
}
